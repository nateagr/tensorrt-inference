import os, sys

import torch
import tensorrt as trt
import pycuda.autoinit # without this, "LogicError: explicit_context_dependent failed: invalid device context - no currently active context?"

from tensorrt_inference.backend import gpu_abbrev


def build_engine_vit(
    output_trt_model_file: str,
    onnx_model_file: str,
    min_batch_size: int,
    optimal_batch_size: int,
    max_batch_size: int,
    precision: str = "fp16",
    log_level = trt.Logger.INFO,
    workspace_size_mbs: int = 32768
):
    trt_version = int(trt.__version__[:3].replace('.','')) # e.g., version 8.4.1.5 becomes 84
    gpu_name = gpu_abbrev(torch.cuda.get_device_name())
    trt_logger = trt.Logger(log_level)
    trt_builder = trt.Builder(trt_logger)

    if os.path.exists(output_trt_model_file):
        print(f'Engine file {output_trt_model_file} exists. Skip building...')
        return

    print(f'Building {precision} engine of {onnx_model_file} model on {gpu_name} GPU...')

    ## parse ONNX model
    network = trt_builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, trt_logger)
    parse_success = onnx_parser.parse_from_file(onnx_model_file)
    for idx in range(onnx_parser.num_errors):
        print(onnx_parser.get_error(idx))
    if not parse_success:
        sys.exit('ONNX model parsing failed')
    
    ## build TRT engine (configuration options at: https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/BuilderConfig.html#ibuilderconfig)
    config = trt_builder.create_builder_config()

    nb_channels = network.get_input(0).shape[1]
    height = network.get_input(0).shape[2]
    width = network.get_input(0).shape[3]
        
    # handle dynamic shape (min/opt/max): https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes
    # by default batch dim set as 1 for all min/opt/max. If there are batch need, change the value for opt and max accordingly
    profile = trt_builder.create_optimization_profile() 
    profile.set_shape(
        "image_features",
        (min_batch_size, nb_channels, height, width),
        (optimal_batch_size, nb_channels, height, width),
        (max_batch_size, nb_channels, height, width)
    ) 
    config.add_optimization_profile(profile)
    
    if trt_version >= 84:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size_mbs * (1 << 20)) # 4096 MiB, syntax after TRT 8.4
    else:
        config.max_workspace_size = workspace_size_mbs * (1 << 20) # syntax before TRT 8.4

    # precision
    if precision == 'fp32':
        config.clear_flag(trt.BuilderFlag.TF32) # TF32 enabled by default, need to clear flag
    elif precision == 'tf32':
        pass
    elif precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)

    # build
    serialized_engine = trt_builder.build_serialized_network(network, config)
    
    ## save TRT engine
    with open(output_trt_model_file, 'wb') as f:
        f.write(serialized_engine)
    print(f'Engine is saved to {output_trt_model_file}')
