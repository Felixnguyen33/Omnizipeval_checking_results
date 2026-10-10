"""Resumable MMMU_DEV_VAL inference with native VLMEvalKit prompts and scoring."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
DATASET = 'MMMU_DEV_VAL'
MIN_PIXELS = 4 * 32 * 32
MAX_PIXELS = 16384 * 32 * 32


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temp.replace(path)


def read_records(path):
    records = {}
    if path.exists():
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
                    raise ValueError(f'Duplicate prediction index: {key}')
                records[key] = record
    return records


def config_override(config):
    if getattr(config, 'model_type', None) == 'qwen3_vl_moe':
        config.text_config.architectures = ['Qwen3MoeForCausalLM']
    return config


def make_engine(args, out, max_images):
    from vllm import LLM
    kwargs = dict(
        model=args.model, dtype='bfloat16', tensor_parallel_size=1,
        pipeline_parallel_size=7, distributed_executor_backend='mp',
        enable_expert_parallel=False, gpu_memory_utilization=0.85,
        max_model_len=32768, max_num_seqs=16, max_num_batched_tokens=8192,
        enforce_eager=True, disable_custom_all_reduce=True, seed=0,
        enable_prefix_caching=True, hf_overrides=config_override,
        limit_mm_per_prompt={'image': max_images, 'video': 0},
        mm_processor_kwargs={'min_pixels': MIN_PIXELS, 'max_pixels': MAX_PIXELS},
    )
    if args.dynamic_int8_moe_fix:
        kwargs['worker_extension_cls'] = 'qwen3_vl_mmmu_int8.DynamicInt8MoEWorker'
    llm = LLM(**kwargs)
    if args.dynamic_int8_moe_fix:
        audit = llm.collective_rpc('audit_dynamic_int8_moe')
        expected = json.loads((Path(args.model) / 'config.json').read_text())['text_config']['num_hidden_layers']
        if sum(map(len, audit)) != expected:
            raise RuntimeError(f'Expected {expected} dynamic W8A8 expert layers')
        save(out / 'quantization_audit.json', audit)
        print('W8A8 AUDIT PASSED', expected, 'expert layers', flush=True)
    return llm


def request_for(processor, message):
    from qwen_vl_utils import process_vision_info
    content = []
    for part in message:
        if part['type'] == 'image':
            content.append(dict(type='image', image=part['value'],
                                min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS))
        elif part['type'] == 'text':
            content.append(dict(type='text', text=part['value']))
        else:
            raise ValueError(part['type'])
    messages = [dict(role='user', content=content)]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, _ = process_vision_info(messages, image_patch_size=16)
    return dict(prompt=text, multi_modal_data={'image': images},
                mm_processor_kwargs={'do_resize': False})


def run(args):
    from transformers import AutoProcessor
    from vlmeval.dataset import build_dataset
    from vlmeval.vlm.qwen3_vl.prompt import Qwen3VLPromptMixin
    from vlmeval.smp import dump, load, get_intermediate_file_path

    out = Path(args.output) / args.name
    out.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(DATASET)
    data = dataset.data.copy()
    source = Path(os.environ['LMUData']) / f'{DATASET}.tsv'
    digest = hashlib.md5(source.read_bytes()).hexdigest()
    if digest != dataset.DATASET_MD5[DATASET]:
        raise ValueError('MMMU dataset checksum mismatch')
    if len(data) != 1050 or data['index'].nunique() != 1050:
        raise ValueError('Expected 1,050 distinct MMMU dev + validation questions')
    prompt = Qwen3VLPromptMixin()
    prompt.dump_image = lambda row, _: dataset.dump_image(row)
    messages = {str(row['index']): prompt.build_prompt(row, DATASET) for _, row in data.iterrows()}
    image_counts = {key: sum(p['type'] == 'image' for p in value) for key, value in messages.items()}
    for message in messages.values():
        for part in message:
            if part['type'] == 'image' and not Path(part['value']).is_file():
                raise FileNotFoundError(part['value'])
    manifest = dict(dataset=DATASET, md5=digest, count=len(data),
                    splits=data['split'].value_counts().to_dict(),
                    image_counts=dict(Counter(image_counts.values())),
                    question_types=data['question_type'].value_counts().to_dict(),
                    indices=[str(i) for i in data['index']])
    save(out / 'dataset_manifest.json', manifest)
    protocol = dict(model=args.model, repo=args.repo, revision=args.revision,
                    dataset=DATASET, dataset_md5=digest, expected=1050,
                    temperature=0.0, top_p=1.0, seed=0, max_tokens=args.max_tokens,
                    min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, max_model_len=32768,
                    prompt='VLMEvalKit Qwen3VLPromptMixin._build_mmmu_prompt',
                    scoring='VLMEvalKit MMMUDataset.evaluate exact_matching',
                    pipeline_parallel_size=7, tensor_parallel_size=1,
                    dynamic_int8_moe_fix=args.dynamic_int8_moe_fix,
                    worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    protocol_path = out / f'{DATASET}.protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Protocol changed; use a new output directory')
    save(protocol_path, protocol)
    if args.preflight_only:
        print('PREFLIGHT PASSED', json.dumps({k:v for k,v in manifest.items() if k != 'indices'}), flush=True)
        return
    checkpoint = out / f'{DATASET}.jsonl'
    records = read_records(checkpoint)
    indices = set(manifest['indices'])
    if not set(records).issubset(indices):
        raise ValueError('Predictions contain indices outside MMMU_DEV_VAL')
    status_path = out / f'{DATASET}.status.json'

    def status(state, **extra):
        save(status_path, dict(status=state, completed=len(records), total=len(data),
                               updated_utc=now(), truncated=sum(r['finish_reason']=='length' for r in records.values()),
                               **extra))

    pending = [row for _, row in data.iterrows() if str(row['index']) not in records]
    if pending:
        # Exercise an ordinary MCQ, an open question, and the largest multi-image input first.
        probes = [str(data.iloc[0]['index']), str(data[data['question_type']=='open'].iloc[0]['index']),
                  max(image_counts, key=image_counts.get)]
        pending.sort(key=lambda row: probes.index(str(row['index'])) if str(row['index']) in probes else 3)
        status('loading')
        processor = AutoProcessor.from_pretrained(args.model)
        llm = make_engine(args, out, max(image_counts.values()))
        from vllm import SamplingParams
        params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, seed=0)
        batches = [pending[:3]] + [pending[i:i+args.batch_size] for i in range(3, len(pending), args.batch_size)]
        for batch in batches:
            requests = [request_for(processor, messages[str(row['index'])]) for row in batch]
            outputs = llm.generate(requests, params, use_tqdm=True)
            if len(outputs) != len(batch):
                raise RuntimeError('Missing generation outputs')
            with checkpoint.open('a') as stream:
                for row, req, output in zip(batch, requests, outputs):
                    answer = output.outputs[0]
                    if not answer.text.strip():
                        raise RuntimeError(f'Empty prediction at index {row["index"]}')
                    if len(output.prompt_token_ids) + args.max_tokens > 32768:
                        raise RuntimeError('Input exceeds the configured context budget')
                    record = dict(index=str(row['index']), id=row['id'], split=row['split'],
                                  prediction=answer.text, finish_reason=answer.finish_reason,
                                  tokens=len(answer.token_ids), prompt_tokens=len(output.prompt_token_ids),
                                  image_count=image_counts[str(row['index'])],
                                  prompt_sha256=hashlib.sha256(req['prompt'].encode()).hexdigest(),
                                  completed_utc=now())
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                    records[record['index']] = record
                stream.flush()
                os.fsync(stream.fileno())
            status('running')
            print('PROGRESS', args.name, f'{len(records)}/{len(data)}', flush=True)
    if set(records) != indices:
        raise ValueError('Incomplete prediction coverage')
    status('scoring')
    data['prediction'] = [records[str(i)]['prediction'] for i in data['index']]
    data['finish_reason'] = [records[str(i)]['finish_reason'] for i in data['index']]
    raw_file = out / f'{args.name}_{DATASET}.tsv'
    dump(data.drop(columns=['image'], errors='ignore'), str(raw_file))
    score = dataset.evaluate(str(raw_file), model='exact_matching', nproc=4)
    save(out / f'{DATASET}.metrics.json', score.to_dict(orient='records'))
    evaluated = load(get_intermediate_file_path(str(raw_file), '_exact_matching_result'))
    if len(evaluated) != 1050 or evaluated['index'].nunique() != 1050:
        raise RuntimeError('Scored rows do not cover the complete dataset')
    summary = dict(overall_accuracy_percent=100*float(evaluated['hit'].mean()),
                   correct=int(evaluated['hit'].sum()), count=len(evaluated),
                   splits={split:dict(count=len(group), correct=int(group['hit'].sum()),
                                     accuracy_percent=100*float(group['hit'].mean()))
                           for split, group in evaluated.groupby('split')},
                   unmatched=int(evaluated['log'].str.contains('Failed in Prefetch', na=False).sum()))
    save(out / 'summary.json', summary)
    status('complete', **summary)
    print('COMPLETE', args.name, json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--dynamic-int8-moe-fix', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    run(parser.parse_args())
