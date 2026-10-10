import os, csv, json
import argparse
import hashlib
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

URL = os.environ.get("LONGBENCH_API_BASE", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("LONGBENCH_API_KEY", "token-abc123")
template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def encode_prompt(tokenizer, prompt, disallowed_special=None):
    if disallowed_special is not None:
        input_ids = tokenizer.encode(prompt, disallowed_special=disallowed_special)
    else:
        input_ids = tokenizer.encode(prompt)
    if hasattr(input_ids, "ids"):
        input_ids = input_ids.ids
    return input_ids

def decode_prompt(tokenizer, input_ids):
    if hasattr(input_ids, "ids"):
        input_ids = input_ids.ids
    try:
        return tokenizer.decode(input_ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(input_ids)

def query_llm(prompt, model, tokenizer, client=None, temperature=0.5, max_new_tokens=128, stop=None,
              top_p=1.0, presence_penalty=0.0, frequency_penalty=0.0,
              max_len=None, served_name=None, hf_tokenizer=None, extra_body=None, seed=None):
    # truncate: if the rendered prompt is longer than max_len tokens, keep the
    # first max_len//2 and the last max_len//2 tokens (upstream middle truncation).
    # max_len / served_name / hf_tokenizer default to the config/*.json registry.
    if max_len is None:
        max_len = maxlen_map[model]
    if hf_tokenizer is None:
        hf_tokenizer = model in model_map
    if hf_tokenizer:
        input_ids = encode_prompt(tokenizer, prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = decode_prompt(tokenizer, input_ids)
    else:
        input_ids = encode_prompt(tokenizer, prompt, disallowed_special=())
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = decode_prompt(tokenizer, input_ids)
    tries = 0
    if served_name is None:
        served_name = model_map.get(model, model)
    # Optional request fields (only sent when set, so the default request is upstream's).
    optional = {}
    if extra_body:
        optional["extra_body"] = extra_body
    if seed is not None:
        optional["seed"] = seed
    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=served_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                **optional,
            )
            return completion.choices[0].message.content
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return ''

def extract_answer(response):
    response = response.replace('*', '')
    # Reasoning models emit a <think>...</think> trace before the final answer;
    # strip it so the regex matches only the post-thought content.
    if '</think>' in response:
        response = response.split('</think>', 1)[1]
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            # Fall back to boxed{X} (DeepSeek-recommended format) and a trailing bare letter
            match = re.search(r'\\boxed\{\(?([A-D])\)?\}', response)
            if match:
                return match.group(1)
            match = re.search(r'\b([A-D])\b(?!.*\b[A-D]\b)', response, re.DOTALL)
            if match:
                return match.group(1)
            return None

def needs_capped_reasoning_budget(model_name):
    # DeepSeek-R1 (and its distills) and Qwen3 share the same recommended
    # thinking-mode sampling (temperature=0.6, top_p=0.95) and, on the
    # smaller checkpoints, the same risk of long repetition loops before
    # emitting a final answer. Cap their output at 4096 tokens instead of
    # the full 32K budget given to bigger/less collapse-prone reasoners
    # (QwQ-32B, o1) so a single stuck generation doesn't blow up wall time
    # -- and, for Qwen3-8B specifically, so there's still room left in its
    # much shorter (40960) native context for the input.
    name = model_name.lower()
    return any(k in name for k in ["r1-distill", "deepseek-r1", "deepseek_llama_", "qwen3"])

def is_reasoning_model(model_name):
    # Any model that emits a <think> trace needs a bigger output budget
    # than the answer-sized 128/1024 used for non-reasoning models.
    name = model_name.lower()
    return any(k in name for k in ["r1-distill", "deepseek-r1", "deepseek_llama_", "o1-", "qwq", "qwen3"])

def uses_openai_tokenizer(model_name):
    # Registered HF models, including SparseGPT/GPTQ, use their own tokenizer.
    return model_name not in model_map and model_name.startswith(("gpt-", "o1"))

def request_seed(seed, uid, stage):
    # Per-request seed derived from (--seed, question id, stage), so a resumed or
    # re-ordered run sends every question the same seed. None = no seed sent.
    if seed is None:
        return None
    return int(hashlib.sha256(f"{seed}:{uid}:{stage}".encode()).hexdigest()[:8], 16) % (2**31)

def generation_settings(args):
    """Effective per-run settings. Model-name heuristics give the defaults;
    every explicit command-line option overrides them."""
    model = args.model
    reasoning = is_reasoning_model(model)
    capped = needs_capped_reasoning_budget(model)
    # DeepSeek R1 / Qwen3 sampling per their model cards: temperature=0.6,
    # top_p=0.95, no system prompt (pred.py only sends a user message anyway).
    temp = 0.6 if capped else 0.1
    top_p = 0.95 if capped else 1.0
    cot_max = (4096 if capped else 32768) if reasoning else 1024
    if reasoning:
        ans_max = 4096 if capped else 32768
    elif args.cot:
        # CoT stage 2 only needs a short \boxed{X} extraction.
        ans_max = 128
    else:
        # 0shot.txt asks for step-by-step reasoning + \boxed{}; 128 tokens
        # truncates before the answer.
        ans_max = 1024
    # --max_new_tokens overrides the auto-selected budget.
    if args.max_new_tokens is not None:
        cot_max = args.max_new_tokens
        ans_max = args.max_new_tokens
    if args.cot_max_new_tokens is not None:
        cot_max = args.cot_max_new_tokens
    if args.ans_max_new_tokens is not None:
        ans_max = args.ans_max_new_tokens
    if args.temperature is not None:
        temp = args.temperature
    if args.top_p is not None:
        top_p = args.top_p
    extra_body = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.repetition_penalty is not None:
        extra_body["repetition_penalty"] = args.repetition_penalty
    if args.enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking == "true"}

    model_path = args.model_path or model_map.get(model)
    openai_tok = args.model_path is None and uses_openai_tokenizer(model)
    return {
        "temperature": temp, "top_p": top_p,
        "cot_max_new_tokens": cot_max, "ans_max_new_tokens": ans_max,
        "extra_body": extra_body, "seed": args.seed,
        "max_len": args.max_len if args.max_len is not None else maxlen_map[model],
        "served_name": args.served_model_name or model_path or model,
        "tokenizer": None if openai_tok else (args.tokenizer or model_path),
        "hf_tokenizer": not openai_tok,
    }

def get_pred(data, args, fout):
    model = args.model
    gen = generation_settings(args)
    if not gen["hf_tokenizer"]:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained(gen["tokenizer"], trust_remote_code=True)
    # Reasoning models can take many minutes on long contexts; default client
    # timeouts are too short and leave partial result files.
    request_timeout = float(os.environ.get("LONGBENCH_REQUEST_TIMEOUT", "600"))
    client = OpenAI(
        base_url=URL,
        api_key=API_KEY,
        timeout=request_timeout,
    )
    common = dict(temperature=gen["temperature"], top_p=gen["top_p"], max_len=gen["max_len"],
                  served_name=gen["served_name"], hf_tokenizer=gen["hf_tokenizer"],
                  extra_body=gen["extra_body"])
    cot_max, ans_max = gen["cot_max_new_tokens"], gen["ans_max_new_tokens"]
    for item in tqdm(data):
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        if args.cot:
            output = query_llm(prompt, model, tokenizer, client, max_new_tokens=cot_max,
                               seed=request_seed(args.seed, item['_id'], 'cot'), **common)
        else:
            output = query_llm(prompt, model, tokenizer, client, max_new_tokens=ans_max,
                               seed=request_seed(args.seed, item['_id'], 'answer'), **common)
        if output == '':
            continue
        if args.cot: # extract answer
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            output = query_llm(prompt, model, tokenizer, client, max_new_tokens=ans_max,
                               seed=request_seed(args.seed, item['_id'], 'answer'), **common)
            if output == '':
                continue
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    print("Effective settings:", json.dumps(generation_settings(args)))
    tok_suffix = f"_t{args.max_new_tokens}" if args.max_new_tokens is not None else ""
    base = args.model.split("/")[-1]
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, f"{base}_rag_{args.rag}{tok_suffix}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, f"{base}_no_context{tok_suffix}.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, f"{base}_cot{tok_suffix}.jsonl")
    else:
        out_file = os.path.join(args.save_dir, f"{base}{tok_suffix}.jsonl")

    dataset = load_dataset('THUDM/LongBench-v2', split='train', revision=args.dataset_revision) # dataset = json.load(open('data.json', 'r', encoding='utf-8'))
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]
    print(f"Loaded {len(data_all)} questions from THUDM/LongBench-v2 (revision={args.dataset_revision or 'latest'})")

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    processes = []
    for rank in range(args.n_proc):
        p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat",
                        help="Registry key in config/*.json, or (with --model_path and --max_len) any run name; "
                             "also names the output file.")
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--max_new_tokens", "-t", type=int, default=None,
                        help="Override per-model auto-budget for output tokens; also suffixes the output filename with _t{N}.")
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    # Explicit protocol options (OmniZipEval). None = the model-name defaults above.
    parser.add_argument("--model_path", type=str, default=None,
                        help="HF id or local path of the served checkpoint (instead of config/model2path.json).")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer used for input truncation (default: --model_path / registry path).")
    parser.add_argument("--served_model_name", type=str, default=None,
                        help="Model name sent to the OpenAI-compatible server (default: --model_path / registry path).")
    parser.add_argument("--max_len", type=int, default=None,
                        help="Input-token cap for middle truncation (default: config/model2maxlen.json).")
    parser.add_argument("--cot_max_new_tokens", type=int, default=None, help="Output budget of the CoT stage (stage 1).")
    parser.add_argument("--ans_max_new_tokens", type=int, default=None, help="Output budget of the answer stage (stage 2 / no-CoT).")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None, help="Sent as vLLM extra_body top_k.")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Sent as vLLM extra_body repetition_penalty.")
    parser.add_argument("--enable_thinking", choices=["true", "false"], default=None,
                        help="Sent as chat_template_kwargs.enable_thinking (Qwen3). Not sent when omitted.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base seed; each request gets sha256(f'{seed}:{_id}:{stage}') mod 2^31. Not sent when omitted.")
    parser.add_argument("--dataset_revision", type=str, default=None,
                        help="Hub revision of THUDM/LongBench-v2 (default: latest).")
    args = parser.parse_args()
    main()
