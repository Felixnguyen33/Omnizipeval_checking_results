try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
except ImportError as e:
    # print("Cannot import vllm")
    pass

from lcb_runner.runner.base_runner import BaseRunner
from lcb_runner.utils.extraction_utils import repair_raw_sentencepiece_text


class VLLMRunner(BaseRunner):
    def __init__(self, args, model):
        super().__init__(args, model)
        model_path = (
            model.model_name if args.local_model_path is None else args.local_model_path
        )
        tokenizer_path = getattr(args, "tokenizer", None) or model_path
        llm_kwargs = dict(
            model=model_path,
            tokenizer=tokenizer_path,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=getattr(args, "pipeline_parallel_size", 1),
            dtype=args.dtype,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            enable_prefix_caching=args.enable_prefix_caching,
            trust_remote_code=args.trust_remote_code,
            gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.9),
        )
        # Without an explicit cap, vLLM uses the HF config max (often 131072 for
        # Qwen/R1), which OOMs KV cache on 24GB GPUs after loading 7B weights.
        if getattr(args, "max_model_len", None) is not None:
            llm_kwargs["max_model_len"] = args.max_model_len
        self.llm = LLM(**llm_kwargs)
        self._hf_tokenizer = self.llm.get_tokenizer()
        stop = self.args.stop
        # Empty / blank stop entries break some vLLM versions; drop them.
        if isinstance(stop, list):
            stop = [s for s in stop if s]
        self.sampling_params = SamplingParams(
            n=self.args.n,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop or None,
        )

    def _decode_output(self, request_output) -> str:
        """Prefer vLLM text; fall back to HF decode / SP piece repair."""
        text = request_output.text or ""
        if text.count("Ġ") >= 3 or text.count("Ċ") >= 3:
            token_ids = getattr(request_output, "token_ids", None)
            if token_ids is not None:
                try:
                    text = self._hf_tokenizer.decode(
                        token_ids, skip_special_tokens=True
                    )
                except Exception:
                    text = repair_raw_sentencepiece_text(text)
            else:
                text = repair_raw_sentencepiece_text(text)
        return text

    def _run_single(self, prompt: str) -> list[str]:
        pass

    def run_batch(self, prompts: list[str]) -> list[list[str]]:
        outputs = [None for _ in prompts]
        remaining_prompts = []
        remaining_indices = []
        for prompt_index, prompt in enumerate(prompts):
            if self.args.use_cache and prompt in self.cache:
                if len(self.cache[prompt]) == self.args.n:
                    # Repair cached outputs that contain raw SP pieces.
                    outputs[prompt_index] = [
                        repair_raw_sentencepiece_text(t) for t in self.cache[prompt]
                    ]
                    continue
            remaining_prompts.append(prompt)
            remaining_indices.append(prompt_index)
        if remaining_prompts:
            vllm_outputs = self.llm.generate(remaining_prompts, self.sampling_params)
            if self.args.use_cache:
                assert len(remaining_prompts) == len(vllm_outputs)
                for index, remaining_prompt, vllm_output in zip(
                    remaining_indices, remaining_prompts, vllm_outputs
                ):
                    texts = [self._decode_output(o) for o in vllm_output.outputs]
                    self.cache[remaining_prompt] = texts
                    outputs[index] = texts
            else:
                for index, vllm_output in zip(remaining_indices, vllm_outputs):
                    outputs[index] = [
                        self._decode_output(o) for o in vllm_output.outputs
                    ]
        return outputs
