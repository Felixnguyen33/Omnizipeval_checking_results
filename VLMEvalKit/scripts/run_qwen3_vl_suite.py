"""Resumable batched Qwen3-VL evaluations using VLMEvalKit datasets and scorers.

Usage: `infer --model <checkpoint> --name <run> --output <dir> [--pp N] [--datasets ...]`,
then `judge --model <judge model> --output <dir>` to label MMSafetyBench answers (see --help).
Raw responses are retained even when GUI coordinates need conversion for scoring.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def save_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, default=str) + '\n')
    temp.replace(path)


def read_records(path):
    records = {}
    if path.exists():
        # Only an interrupted final append may be incomplete.
        with path.open('r+b') as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    stream.truncate(offset)
                    break
                record = json.loads(line)
                key = str(record['index'])
                if key in records:
                    raise ValueError(f'Duplicate checkpoint index: {key}')
                records[key] = record
    return records


def parse_point(response):
    """Qwen3 coordinates are in [0,1000], independently of image dimensions."""
    match = re.search(r'"point_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', response)
    if match is None:
        return None
    x, y = map(float, match.groups())
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return [x / 1000, y / 1000]


def gui_prediction(response):
    point = parse_point(response)
    if point is None:
        # A deliberately out-of-bounds point cannot accidentally hit the origin.
        return 'pyautogui.click(x=1000000000, y=1000000000)'
    return f'pyautogui.click(x={point[0]:.9f}, y={point[1]:.9f})'


def qwen3_vl_config(config):
    # vLLM 0.27 revalidates the nested language model config, whose HF
    # checkpoint omits architectures. Supply its existing language architecture.
    if getattr(config, 'model_type', None) == 'qwen3_vl_moe':
        config.text_config.architectures = ['Qwen3MoeForCausalLM']
    return config


def engine(model, args, multimodal=True):
    from vllm import LLM
    kwargs = dict(
        model=model, dtype='bfloat16', tensor_parallel_size=1,
        pipeline_parallel_size=args.pp, distributed_executor_backend='mp',
        enable_expert_parallel=False, gpu_memory_utilization=0.85,
        max_model_len=32768, max_num_seqs=32, max_num_batched_tokens=8192,
        enforce_eager=True, disable_custom_all_reduce=True, seed=0,
        enable_prefix_caching=True,
    )
    if multimodal:
        kwargs.update(hf_overrides=qwen3_vl_config,
                      limit_mm_per_prompt={'image': 1, 'video': 0},
                      mm_processor_kwargs={'min_pixels': 4 * 32 * 32,
                                           'max_pixels': 16384 * 32 * 32})
    if args.dynamic_int8_moe_fix:
        kwargs['worker_extension_cls'] = 'qwen3_vl_int8_runtime.DynamicInt8MoEWorker'
    llm = LLM(**kwargs)
    if args.dynamic_int8_moe_fix:
        audited = llm.collective_rpc('audit_dynamic_int8_moe')
        expected = json.loads((Path(model) / 'config.json').read_text())['text_config']['num_hidden_layers']
        if sum(map(len, audited)) != expected:
            raise RuntimeError(f'Expected {expected} W8A8 expert layers, got {audited}')
        save_json(Path(args.output) / args.name / 'quantization_audit.json', audited)
        print(f'QUANTIZATION AUDIT: {expected} expert layers use INT8 weights and dynamic per-token INT8 activations', flush=True)
    return llm


def dataset_prompt(dataset, row, name):
    from vlmeval.vlm.qwen3_vl.prompt import Qwen3VLPromptMixin
    if name == 'ScreenSpot_v2':
        images = dataset.dump_image(row)
        image_root = Path(dataset.dataset_map[row['SUB_DATASET']].img_root)
        images = [str(Path(p) if Path(p).is_absolute() else image_root / p) for p in images]
        if not all(Path(p).is_file() for p in images):
            raise FileNotFoundError(images)
        question = (
            f'Locate the UI element described as: {row["question"]}\n'
            'Return the center of the element as JSON: {"point_2d": [x, y]}. '
            'Use integer coordinates normalized to the range 0 to 1000. '
            'Return only the JSON.'
        )
        return [dict(type='image', value=p) for p in images] + [dict(type='text', value=question)]
    if name == 'MMSafetyBench':
        return dataset.build_prompt(row)
    prompt = Qwen3VLPromptMixin()
    prompt.dump_image = lambda line, _: dataset.dump_image(line)
    return prompt.build_prompt(row, name)


def request_for(processor, message):
    from qwen_vl_utils import process_vision_info
    contents = []
    for part in message:
        if part['type'] == 'image':
            contents.append(dict(type='image', image=part['value'],
                                 min_pixels=4 * 32 * 32, max_pixels=16384 * 32 * 32))
        elif part['type'] == 'text':
            contents.append(dict(type='text', text=part['value']))
        else:
            raise ValueError(part['type'])
    messages = [dict(role='user', content=contents)]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, _ = process_vision_info(messages, image_patch_size=16)
    return dict(prompt=text, multi_modal_data={'image': images},
                mm_processor_kwargs={'do_resize': False})


def infer(args):
    from transformers import AutoProcessor
    from vllm import SamplingParams
    from vlmeval.dataset import build_dataset
    from vlmeval.smp import dump

    out = Path(args.output) / args.name
    out.mkdir(parents=True, exist_ok=True)
    datasets = {name: build_dataset(name) for name in args.datasets}
    processor = AutoProcessor.from_pretrained(args.model)
    llm = None
    for name, dataset in datasets.items():
        data = dataset.data.copy()
        if args.limit:
            data = data.iloc[:args.limit].copy()
        status_path = out / f'{name}.status.json'
        checkpoint = out / f'{name}.jsonl'
        records = read_records(checkpoint)
        indices = {str(x) for x in data['index']}
        if not set(records).issubset(indices):
            raise ValueError('Checkpoint contains indices outside this dataset/split.')
        protocol = dict(model=args.model, name=args.name, dataset=name, count=len(data),
                        temperature=0.0, seed=0, max_pixels=16384*32*32,
                        max_tokens=2048 if name == 'MMSafetyBench' else 128,
                        prompt='qwen3_point_2d_v1' if name == 'ScreenSpot_v2' else 'vlmevalkit',
                        pipeline_parallel_size=args.pp, tensor_parallel_size=1)
        if args.dynamic_int8_moe_fix:
            protocol['runtime_fix'] = 'ct_dynamic_int8_moe_v1'
        protocol_path = out / f'{name}.protocol.json'
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError(f'Protocol changed; choose a new output directory: {protocol_path}')
        save_json(protocol_path, protocol)
        pending = [row for _, row in data.iterrows() if str(row['index']) not in records]
        save_json(status_path, dict(status='loading' if pending else 'scoring', completed=len(records), total=len(data)))
        if pending and llm is None:
            llm = engine(args.model, args)
        params = SamplingParams(temperature=0.0, max_tokens=protocol['max_tokens'], seed=0)
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            requests = [request_for(processor, dataset_prompt(dataset, row, name)) for row in batch]
            outputs = llm.generate(requests, params, use_tqdm=True)
            with checkpoint.open('a') as stream:
                for row, req, output in zip(batch, requests, outputs):
                    answer = output.outputs[0]
                    if not answer.text.strip():
                        raise RuntimeError(f'Empty prediction for {name}/{row["index"]}')
                    record = dict(index=str(row['index']), prediction=answer.text,
                                  finish_reason=answer.finish_reason, tokens=len(answer.token_ids),
                                  prompt_sha256=hashlib.sha256(req['prompt'].encode()).hexdigest())
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                    records[record['index']] = record
                stream.flush()
                os.fsync(stream.fileno())
            save_json(status_path, dict(status='running', completed=len(records), total=len(data), updated=time.time()))
            print(f'PROGRESS {args.name} {name}: {len(records)}/{len(data)}', flush=True)
        assert set(records) == indices
        data['prediction'] = [records[str(i)]['prediction'] for i in data['index']]
        data['finish_reason'] = [records[str(i)]['finish_reason'] for i in data['index']]
        data = data.drop(columns=['image'], errors='ignore')
        raw_file = out / f'{args.name}_{name}.tsv'
        dump(data, str(raw_file))
        extra = dict(truncated=sum(r['finish_reason'] == 'length' for r in records.values()))
        if name == 'MMSafetyBench':
            save_json(status_path, dict(status='awaiting_judge', completed=len(data), total=len(data), **extra))
            continue
        if name == 'ScreenSpot_v2':
            score_data = data.copy()
            score_data['raw_prediction'] = score_data['prediction']
            score_data['prediction'] = [gui_prediction(s) for s in data['prediction']]
            score_file = out / f'{args.name}_{name}_normalized.tsv'
            dump(score_data, str(score_file))
            score = dataset.evaluate(str(score_file), model='exact_matching', nproc=4)
            counts = data.groupby('SUB_DATASET').size().to_dict()
            overall = sum(score[f'{key}:Overall_Accuracy'] * n for key, n in counts.items()) / len(data)
            extra.update(overall_accuracy=overall,
                         format_errors=sum(parse_point(s) is None for s in data['prediction']))
            score.update(Overall_Accuracy=overall, Format_Err_Rate=100*extra['format_errors']/len(data))
        else:
            score = dataset.evaluate(str(raw_file), model='exact_matching', nproc=4)
            score = score.to_dict(orient='records')
        save_json(out / f'{name}.metrics.json', score)
        save_json(status_path, dict(status='complete', completed=len(data), total=len(data), **extra))
        print('SCORE', name, json.dumps(score), flush=True)


def judge_prompt(prediction, category):
    from vlmeval.dataset.mmsafetybench import (GENERAL_SAFETY_TEMPLATE, POLITICAL_LOBBYING_TEMPLATE,
                                             LEGAL_FIN_HEALTH_TEMPLATE, GOV_DECISION_TEMPLATE)
    if category == '08-Political_Lobbying':
        template = POLITICAL_LOBBYING_TEMPLATE
    elif category in ['10-Legal_Opinion', '11-Financial_Advice', '12-Health_Consultation']:
        template = LEGAL_FIN_HEALTH_TEMPLATE
    elif category == '13-Gov_Decision':
        template = GOV_DECISION_TEMPLATE
    else:
        template = GENERAL_SAFETY_TEMPLATE
    return template.format(prediction=prediction)


def judge(args):
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vlmeval.dataset import build_dataset
    from vlmeval.dataset.mmsafetybench import _normalize_label
    from vlmeval.smp import dump, load

    llm = None
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for raw_file in sorted(Path(args.output).glob('*/*_MMSafetyBench.tsv')):
        folder = raw_file.parent
        state_file = folder / 'MMSafetyBench.status.json'
        if state_file.exists() and json.loads(state_file.read_text()).get('status') == 'complete':
            continue
        data = load(str(raw_file))
        checkpoint = folder / 'MMSafetyBench.local_judge.jsonl'
        records = read_records(checkpoint)
        rows = [row for _, row in data.iterrows() if str(row['index']) not in records]
        if rows and llm is None:
            llm = engine(args.model, args, multimodal=False)
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            prompts = [tokenizer.apply_chat_template(
                [dict(role='user', content=judge_prompt(row['prediction'], row['category']))],
                tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in batch]
            outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=128, seed=0), use_tqdm=True)
            with checkpoint.open('a') as stream:
                for row, result in zip(batch, outputs):
                    text = result.outputs[0].text
                    label = _normalize_label(text)
                    if label == 'unknown':
                        raise RuntimeError('Local safety judge produced an unparseable label; no safety score reported.')
                    record = dict(index=str(row['index']), label=label, judge_output=text, judge_model=args.model)
                    records[record['index']] = record
                    stream.write(json.dumps(record) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            save_json(state_file, dict(status='judging', completed=len(data), total=len(data), judged=len(records)))
        assert len(records) == len(data)
        data['original_prediction'] = data['prediction']
        data['prediction'] = [records[str(i)]['label'] for i in data['index']]
        data['judge_response'] = [records[str(i)]['judge_output'] for i in data['index']]
        data['judge_model'] = args.model
        labeled = folder / (raw_file.stem + '_local_judge_labels.tsv')
        dump(data, str(labeled))
        # Exact parsing is used only on the judge's explicit labels, never on model answers.
        scores = build_dataset('MMSafetyBench').evaluate(str(labeled), model='exact_matching')
        scores['judge'] = 'local:' + args.model
        dump(scores, str(folder / 'MMSafetyBench.local_judge_scores.csv'))
        save_json(folder / 'MMSafetyBench.metrics.json', scores.to_dict(orient='records'))
        save_json(state_file, dict(status='complete', completed=len(data), total=len(data),
                                  judge=args.model, protocol='VLMEvalKit local judge; not official GPT-4 scoring'))
        print('SAFETY SCORED', folder.name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['infer', 'judge'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--name', default='judge')
    parser.add_argument('--output', required=True)
    parser.add_argument('--pp', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--dynamic-int8-moe-fix', action='store_true',
                        help='Preserve dynamic W8A8 expert activations in vLLM 0.27.1 and check all expert layers.')
    parser.add_argument('--datasets', nargs='+', default=['RealWorldQA', 'MMBench_DEV_EN_V11', 'ScreenSpot_v2', 'MMSafetyBench'])
    args = parser.parse_args()
    (infer if args.mode == 'infer' else judge)(args)


if __name__ == '__main__':
    main()
