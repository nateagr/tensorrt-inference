import os
import tempfile

import torch
import onnx
from onnx import TensorProto
import onnx_graphsurgeon as gs


def convert_to_onnx(
    output_file: str,
    model: torch.nn.Module,
    seq_len: int = 77,
    batch_size: int = 512
):
    model = model.eval()
    vocab_size = model.config.vocab_size
    device = torch.device('cpu')
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
    attention_mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.long, device=device)
    input_names = ['input_ids', 'attention_mask']   
    output_names = ['output']
    dynamic_axes={'input_ids'   : {0 : 'batch_size'}, 
                'attention_mask'   : {0 : 'batch_size'},   
                'output' : {0 : 'batch_size'}}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_onnx_file = os.path.join(tmp, "tmp.onnx")
        torch.onnx.export(model, # model 
                        (input_ids, attention_mask), # model inputs
                        tmp_onnx_file,
                        export_params=True,
                        opset_version=13,
                        do_constant_folding=True,
                        input_names = input_names,
                        output_names = output_names,
                        dynamic_axes = dynamic_axes)

        graph = gs.import_onnx(onnx.load(tmp_onnx_file))
        graph = _remove_uint8_cast(graph)
        graph.cleanup().toposort()
        onnx.save_model(gs.export_onnx(graph), output_file)
        _check_model(output_file)


def _remove_uint8_cast(graph):
    '''
    Remove all uint8 Cast nodes since TRT doesn't support UINT8 cast op.
    Ref: https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon/examples/06_removing_nodes
    '''
    nodes = [node for node in graph.nodes if node.op == 'Cast' and node.attrs["to"] == TensorProto.UINT8] # find by op name and attribute

    for node in nodes:
        # [ONNX's Cast operator](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast) will exactly have 1 input and 1 output
        # reconnect tensors
        input_node = node.i()
        input_node.outputs = node.outputs
        node.outputs.clear()

        # an alternative way is to just not cast to uint8
        # node.attrs["to"] = TensorProto.INT64

    return graph

        
def _check_model(model_name):
    # Load the ONNX model
    model = onnx.load(model_name)

    # Check that the model is well formed
    onnx.checker.check_model(model)
