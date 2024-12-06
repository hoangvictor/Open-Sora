
import os
import torch
from torch import nn

from vllm.model_executor.layers.quantization.utils.w8a8_utils import apply_fp8_linear, apply_int8_linear

base_dir = os.path.join(os.path.dirname(__file__), '..', '..')


class QuantLinear(nn.Module):

    def __init__(
        self,
        module: nn.Linear,
        old_module_name: str = None,
        quant_mode: str = 'fp8',
    ):
        super().__init__()        

        max_abs_val = torch.max(torch.abs(module.weight.T), axis=0).values.to(torch.float32)

        self.fp8_data_type = torch.float8_e4m3fn
        if torch.version.hip is not None:
            self.fp8_data_type = torch.float8_e4m3fnuz

        self.in_features = module.in_features
        self.out_features = module.out_features
        finfo = torch.finfo(self.fp8_data_type)
        iinfo = torch.iinfo(torch.int8)
        
        assert quant_mode in ['fp8', 'int8']

        self.quant_mode = quant_mode
        self.old_module_name = old_module_name
        
        self.input_scale = None # activations_scales[self.old_module_name].to(torch.float32).unsqueeze(0) # torch.max(activations_scales[self.old_module_name].to(torch.float32)).unsqueeze(0) # None
        if self.input_scale is not None:
            if quant_mode == 'fp8':
                self.input_scale= (self.input_scale / finfo.max)
            else:
                self.input_scale= (self.input_scale / iinfo.max)
            
        if self.quant_mode == 'fp8':
            self.weight_scale = (max_abs_val /  finfo.max)
            self.weight_scale = self.weight_scale.contiguous()
            self.weight = (module.weight.T / self.weight_scale).clamp(min=finfo.min, max=finfo.max).to(self.fp8_data_type)
        elif self.quant_mode == 'int8':
            self.weight_scale = (max_abs_val / iinfo.max)
            self.weight_scale = self.weight_scale.contiguous()
            self.weight = (module.weight.T / self.weight_scale).clamp(min=iinfo.min, max=iinfo.max).to(torch.int8)

        if module._parameters['bias'] is not None:
            self.bias = module.bias
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_2d = input.reshape(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], self.weight.shape[1]]
        scales = self.input_scale.repeat(input.shape[0], 1) if self.input_scale is not None else None
        if self.quant_mode == 'fp8':
            output = apply_fp8_linear(input_2d, self.weight, input_scale=scales, weight_scale=self.weight_scale, bias=self.bias, use_per_token_if_dynamic=True)
        elif self.quant_mode == 'int8':
            output = apply_int8_linear(input_2d, self.weight, input_scale=scales, weight_scale=self.weight_scale, bias=self.bias)
        else:
            raise Exception(f"The forward pass of this quant mode `{self.quant_mode}` is not implemented yet.")
        return output.view(*output_shape)

    def extra_repr(self) -> str:
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, bias={self.bias is not None}, quant_mode={self.quant_mode}"