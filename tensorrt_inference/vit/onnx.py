import os
import tempfile

import torch
import onnx
import onnx_graphsurgeon as gs

from tensorrt_inference.onnx import check_model


def convert_to_onnx(
    output_file: str,
    model: torch.nn.Module,
    batch_size: int = 512
):
    model = model.eval()
    device = torch.device('cpu')
    input_names = ['image_features']
    output_names = ["ouput"]
    dynamic_axes={'image_features'   : {0 : 'batch_size'}}
    image_features = torch.randn((batch_size, 3) + model.image_size, device=device)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_onnx_file = os.path.join(tmp, "tmp.onnx")
        torch.onnx.export(model, # model 
                        (image_features,), # model inputs
                        tmp_onnx_file,
                        export_params=True,
                        opset_version=17,
                        do_constant_folding=True,
                        input_names = input_names,
                        output_names = output_names,
                        dynamic_axes = dynamic_axes)

        graph = gs.import_onnx(onnx.load(tmp_onnx_file))
        graph.cleanup().toposort()
        onnx.save_model(gs.export_onnx(graph), output_file)
        check_model(output_file)
