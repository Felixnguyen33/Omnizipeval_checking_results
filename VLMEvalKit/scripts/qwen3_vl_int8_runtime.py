"""Keep dynamic INT8 activations enabled for the vLLM 0.27.1 CT MoE path.

That version's INT8 oracle treats absent static activation scales as W8A16,
even when the compressed-tensors scheme explicitly requests per-token W8A8.
This worker extension only corrects that scheme and checks the loaded kernels.
No installed package or checkpoint files are modified.
"""
import torch
from vllm.model_executor.layers.fused_moe.config import int8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.oracle.int8 import Int8MoeBackend
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 import (  # noqa: E501
    CompressedTensorsW8A8Int8MoEMethod,
)


_original_config = CompressedTensorsW8A8Int8MoEMethod.get_fused_moe_quant_config


def dynamic_int8_config(self, layer):
    config = _original_config(self, layer)
    if (self.int8_backend == Int8MoeBackend.TRITON
            and self.input_quant.dynamic
            and self.input_quant.strategy == 'token'
            and config.quant_dtype is None):
        config = int8_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=None, a2_scale=None,
            per_act_token_quant=True,
        )
    if not (config.quant_dtype == torch.int8 and config.per_act_token_quant):
        raise RuntimeError('The requested CT W8A8 MoE must quantize activations per token.')
    return config


CompressedTensorsW8A8Int8MoEMethod.get_fused_moe_quant_config = dynamic_int8_config


class DynamicInt8MoEWorker:
    def audit_dynamic_int8_moe(self):
        records = []
        for name, layer in self.model_runner.model.named_modules():
            method = getattr(layer, 'quant_method', None)
            if not isinstance(method, CompressedTensorsW8A8Int8MoEMethod):
                continue
            config = method.moe_kernel.fused_experts.quant_config
            valid = (config.use_int8_w8a8 and config.per_act_token_quant
                     and layer.w13_weight.dtype == torch.int8
                     and layer.w2_weight.dtype == torch.int8)
            if not valid:
                raise RuntimeError(f'Loaded expert kernel is not dynamic W8A8: {name}')
            records.append(dict(layer=name, kernel=type(method.moe_kernel.fused_experts).__name__,
                                weights='int8', activations='int8_dynamic_per_token'))
        return records
