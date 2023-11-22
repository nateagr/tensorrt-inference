from typing import Callable, Dict, List, Tuple

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from numpy import ndarray
import pycuda.autoinit
from pycuda._driver import DeviceAllocation, Stream
from tensorrt import (
    ICudaEngine,
    IExecutionContext,
    IBuilderConfig,
    IElementWiseLayer,
    ILayer,
    INetworkDefinition,
    IOptimizationProfile,
    IReduceLayer,
    Logger,
    Runtime,
    DataType
)


# Mapping between TensorRT data types and numpy data types
_trt_type_to_numpy_types = {
    DataType.BOOL: np.bool,
    DataType.FLOAT: np.float32,
    DataType.HALF: np.float16,
    DataType.INT32: np.int32,
    DataType.INT8: np.int8
}


def gpu_abbrev(name):
    '''
    Map GPU device query name to abbreviation.
    
    ::param str name Device name from torch.cuda.get_device_name().
    ::return str GPU abbreviation.
    ''' 

    GPU_LIST = [
        'V100',
        'TITAN',
        'T4',
        'A100',
        'A10G',
        'A10'
    ] 
    # Partial list, can be extended. The order of A100, A10G, A10 matters. They're put in a way to not detect substring A10 as A100
    
    for i in GPU_LIST:
        if i in name:
            return i 
    
    return 'GPU' # for names not in the partial list, use 'GPU' as default


def build_engine(
    runtime: Runtime,
    onnx_file_path: str,
    logger: Logger,
    min_shape: Tuple[int, int],
    optimal_shape: Tuple[int, int],
    max_shape: Tuple[int, int],
    workspace_size: int,
    fp16: bool,
    int8: bool,
    calibrator: trt.IInt8Calibrator = None
) -> ICudaEngine:
    """
    Convert ONNX model to TensorRT engine.
    
    :param runtime: TensorRT runtime used for inference calls / model building
    :param onnx_file_path: path to the ONNX model
    :param logger: specific logger to TensorRT
    :param min_shape: minimal shape of input tensors
    :param optimal_shape: optimal shape of input tensors
    :param max_shape: maximal shape of input tensors
    :param workspace_size: GPU memory to use during model building
    :param fp16: enable FP16 precision
    :param int8: enable INT-8 quantization
    :param calibrator: calibrator to use for INT-8 quantization
    :return: TensorRT engine to run inference
    """
    with trt.Builder(logger) as builder:
        with builder.create_network(
            # Explicit batch mode: all dimensions are explicit and can be dynamic
            flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        ) as network_definition:
            with trt.OnnxParser(network_definition, logger) as parser:
                builder.max_batch_size = max_shape[0]
                config: IBuilderConfig = builder.create_builder_config()
                config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
                config.max_workspace_size = workspace_size
                # enable CUDNN, CUBLAS and CUBLAS_LT
                
                config.set_tactic_sources(
                    tactic_sources=1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUBLAS_LT) | 1 << int(trt.TacticSource.CUDNN)
                )
                
                if int8:
                    assert calibrator is not None, "Calibration is required for int8 quantization"
                    config.set_flag(trt.BuilderFlag.INT8)
                    config.int8_calibrator = calibrator
                if fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
                
                # Ask the builder to prefer the type constraints/hints when choosing layer implementations
                # insteaf of choosing the fastest
                config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
                
                with open(onnx_file_path, "rb") as f:
                    if not parser.parse(f.read()):
                        print(parser.get_error(0))
                
                # The builder selects the kernels that result in the lowest runtime for the optimum
                # input tensor dimensions, and are valid for all input tensor sizes in the valid range
                # between minimum and maximum dimensions
                # At least one optimization profile is required with dynamically resizable inputs
                profile: IOptimizationProfile = builder.create_optimization_profile()
                for num_input in range(network_definition.num_inputs):
                    profile.set_shape(
                        input=network_definition.get_input(num_input).name,
                        min=min_shape,
                        opt=optimal_shape,
                        max=max_shape,
                    )
                config.add_optimization_profile(profile)

                if fp16:
                    # Noticeable differences have been observed when converting some layers in FP16
                    # Force those layers in FP32
                    network_definition = _fix_fp16_network(network_definition)
                trt_engine = builder.build_serialized_network(network_definition, config)
                engine: ICudaEngine = runtime.deserialize_cuda_engine(trt_engine)
                return engine


def save_engine(engine: ICudaEngine, engine_file_path: str) -> None:
    """
    Save a TensorRT engine

    :param engine: TensorRT engine
    :param engine_file_path: output path
    """
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())


def load_engine(
    runtime: Runtime, engine_file_path: str, profile_index: int = 0
) -> Callable[[Dict[str, np.ndarray]], List[np.ndarray]]:
    """
    Load a TensorRT engine
    
    :param runtime: TensorRT runtime
    :param engine_file_path: path to the serialized engine
    :param profile_index: optimization profile index
    :return: function to run inference
    """
    with open(file=engine_file_path, mode="rb") as f:
        engine: ICudaEngine = runtime.deserialize_cuda_engine(f.read())
        return to_inference_fn(engine, profile_index)


def to_inference_fn(
    engine: ICudaEngine, profile_index: int = 0
) -> Callable[[Dict[str, np.ndarray]], List[np.ndarray]]:
    """
    Returns a function to run inference

    :param engine: TensorRT engine
    :param profile_index: optimization profile index
    :return: function to run inference
    """
    context: IExecutionContext = engine.create_execution_context()
    stream: Stream = cuda.Stream()
    context.set_optimization_profile_async(profile_index=profile_index, stream_handle=stream.handle)
    input_binding_indices, output_binding_indices = _get_input_output_binding_indices(engine, profile_index)

    def _inference_fn(inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        return _run_inference(
            context=context,
            host_inputs=inputs,
            input_binding_indices=input_binding_indices,
            output_binding_indices=output_binding_indices,
            stream=stream,
        )

    return _inference_fn


def _fix_fp16_network(network_definition: INetworkDefinition) -> INetworkDefinition:
    """
    Some of the layers should not be quantized because they generate values with very large quantization
    errors when quantized. We force these layer in FP32.
    
    :param network_definition: network definition
    :return: patched network definition
    """
    for layer_index in range(network_definition.num_layers - 1):
        layer: ILayer = network_definition.get_layer(layer_index)
        next_layer: ILayer = network_definition.get_layer(layer_index + 1)
        if layer.type == trt.LayerType.ELEMENTWISE and next_layer.type == trt.LayerType.REDUCE:
            # Cast to access op attributes
            layer.__class__ = IElementWiseLayer
            next_layer.__class__ = IReduceLayer
            if layer.op == trt.ElementWiseOperation.POW:
                layer.precision = trt.DataType.FLOAT
                next_layer.precision = trt.DataType.FLOAT
            layer.set_output_type(index=0, dtype=trt.DataType.FLOAT)
            next_layer.set_output_type(index=0, dtype=trt.DataType.FLOAT)
    return network_definition


def _get_input_output_binding_indices(engine: trt.ICudaEngine, profile_index: int):
    """
    Returns the input/output binding indices for a given optization profile
    https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt_profiles_bindings
    
    :param engine: TensorRT engine
    :param profile_index: optimization profile index
    :return: input and output binding indices
    """
    num_bindings_per_profile = engine.num_bindings // engine.num_optimization_profiles
    start_binding = profile_index * num_bindings_per_profile
    end_binding = start_binding + num_bindings_per_profile
    input_binding_indices: List[int] = []
    output_binding_indices: List[int] = []
    for binding_index in range(start_binding, end_binding):
        if engine.binding_is_input(binding_index):
            input_binding_indices.append(binding_index)
        else:
            output_binding_indices.append(binding_index)
    return input_binding_indices, output_binding_indices


def _run_inference(
    context: IExecutionContext,
    host_inputs: Dict[str, np.ndarray],
    input_binding_indices: List[int],
    output_binding_indices: List[int],
    stream: Stream,
) -> List[np.ndarray]:
    """
    Run inference with TensorRT

    :param context: execution context used to run inference
    :param inputs: model input(s)
    :param input_binding_indices: input binding indices
    :param output_binding_indices: output binding indices
    :param stream: GPU stream to synchronize memories
    :return: model output(s)
    """
    input_list: List[ndarray] = list()
    device_inputs: List[DeviceAllocation] = list()
    for input in host_inputs.values():
        if input.dtype == "int64":
            input: np.ndarray = np.asarray(input, dtype=np.int32)
        input_list.append(input)
        device_input: DeviceAllocation = cuda.mem_alloc(input.nbytes)
        device_inputs.append(device_input)
        cuda.memcpy_htod_async(device_input, input.ravel(), stream)
    _set_input_shapes(context, input_list, input_binding_indices)
    host_placeholders, device_placeholders = _create_output_placeholders(context, output_binding_indices)
    bindings = device_inputs + device_placeholders
    assert context.execute_async_v2(bindings, stream_handle=stream.handle), "failure during inference"
    for host_placeholder, device_placeholder in zip(host_placeholders, device_placeholders):
        cuda.memcpy_dtoh_async(host_placeholder, device_placeholder)
    stream.synchronize()
    return host_placeholders


def _set_input_shapes(
    context: trt.IExecutionContext,
    host_inputs: List[np.ndarray],
    input_binding_indices: List[int]
) -> None:
    """
    Set input shapes. As a result, TensorRT will compute output shapes

    :param context: TensorRT context used to run inference
    :param host_inputs: model inputs
    :param input_binding_indices: input binding indices
    """
    for host_input, binding_index in zip(host_inputs, input_binding_indices):
        context.set_binding_shape(binding_index, host_input.shape)


def _create_output_placeholders(
    context: trt.IExecutionContext,
    output_binding_indices: List[int],
) -> Tuple[List[np.ndarray], List[DeviceAllocation]]:
    """
    Create placeholders for model outputs on host and device

    :param context: TensorRT context used to run inference
    :param output_binding_indices: output binding indices
    :return: placeholders for model outputs on host and device
    """
    assert context.all_binding_shapes_specified
    host_placeholders: List[np.ndarray] = []
    device_placeholders: List[DeviceAllocation] = []
    for binding_index in output_binding_indices:
        host_placeholder = np.empty(
            context.get_binding_shape(binding_index),
            dtype=_trt_type_to_numpy_types[context.engine.get_binding_dtype(binding_index)]
        )
        host_placeholders.append(host_placeholder)
        device_placeholders.append(cuda.mem_alloc(host_placeholder.nbytes))
    return host_placeholders, device_placeholders
