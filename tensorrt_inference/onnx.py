from typing import Dict

import torch
from torch import nn
import onnx


def convert_to_onnx(
    pytorch_model: nn.Module,
    onnx_output_path: str,
    pytorch_inputs: Dict[str, torch.Tensor],
    quantization: bool
) -> None:
    if quantization:
        try:
            from pytorch_quantization.nn import TensorQuantizer
        except ImportError:
            raise ImportError(
                "It seems that pytorch-quantization is not yet installed. "
                "pytorch-quantization is required when you enable quantization"
                "Please find installation instructions on "
                "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization or run:\n"
                "pip3 install git+ssh://git@github.com/NVIDIA/TensorRT#egg=pytorch-quantization\\&"
                "subdirectory=tools/pytorch-quantization/"
            )

        TensorQuantizer.use_fb_fake_quant = True

    dynamic_axis = dict()
    for k in pytorch_inputs.keys():
        dynamic_axis[k] = {0: "batch_size", 1: "sequence"}
    dynamic_axis["output"] = {0: "batch_size"}
    with torch.no_grad():
        torch.onnx.export(
            pytorch_model,
            args=tuple(pytorch_inputs.values()),
            f=onnx_output_path,
            opset_version=12,
            do_constant_folding=True,
            input_names=list(pytorch_inputs.keys()),
            output_names=["output"],
            dynamic_axes=dynamic_axis,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False,
        )
    if quantization:
        TensorQuantizer.use_fb_fake_quant = False
