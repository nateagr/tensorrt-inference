import os, sys

import torch
import tensorrt as trt
import numpy as np
import pycuda.autoinit # without this, "LogicError: explicit_context_dependent failed: invalid device context - no currently active context?"
import pycuda.driver as cuda


class TRTModel:
    '''
    Generic class to run a TRT engine by specifying engine path and giving input data.
    '''
    class HostDeviceMem(object):
        '''
        Helper class to record host-device memory pointer pairs
        '''
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()

    def __init__(self, engine_path):
        self.engine_path = engine_path 
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        # load and deserialize TRT engine
        self.engine = self.load_engine()

        # allocate input/output memory buffers
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

        # create context
        self.context = self.engine.create_execution_context()

        # Dict of NumPy dtype -> torch dtype (when the correspondence exists). From: https://github.com/pytorch/pytorch/blob/e180ca652f8a38c479a3eff1080efe69cbc11621/torch/testing/_internal/common_utils.py#L349
        self.numpy_to_torch_dtype_dict = {
            bool       : torch.bool,
            np.uint8      : torch.uint8,
            np.int8       : torch.int8,
            np.int16      : torch.int16,
            np.int32      : torch.int32,
            np.int64      : torch.int64,
            np.float16    : torch.float16,
            np.float32    : torch.float32,
            np.float64    : torch.float64,
            np.complex64  : torch.complex64,
            np.complex128 : torch.complex128
        }

    def load_engine(self):
        with open(self.engine_path, 'rb') as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine
    
    def allocate_buffers(self, engine):
        '''
        Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
        '''
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in engine: # binding is the name of input/output
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype) # page-locked memory buffer (won't swapped to disk)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer address to device bindings. When cast to int, it's a linear index into the context's memory (like memory address). See https://documen.tician.de/pycuda/driver.html#pycuda.driver.DeviceAllocation
            bindings.append(int(device_mem))

            # Append to the appropriate input/output list.
            if engine.binding_is_input(binding):
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def __call__(self, model_inputs: list):
        '''
        Inference step (like forward() in PyTorch).
        model_inputs: list of numpy array or list of torch.Tensor (on GPU)
        '''
        batch_size = np.unique(np.array([i.size(dim=0) for i in model_inputs]))
        assert len(batch_size) == 1, 'Input batch sizes are not consistent!'
        batch_size = batch_size[0]

        for i, model_input in enumerate(model_inputs):
            binding_name = self.engine[i] # i-th input/output name
            binding_dtype = trt.nptype(self.engine.get_binding_dtype(binding_name)) # trt can only tell to numpy dtype
            model_input = model_input.to(self.numpy_to_torch_dtype_dict[binding_dtype])
            cuda.memcpy_dtod_async(self.inputs[i].device, model_input.data_ptr(), model_input.element_size() * model_input.nelement(), self.stream) # dtod need size in bytes

        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle) # v2 no need for batch_size arg
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        self.stream.synchronize()
        return [torch.from_numpy(out.host.reshape(batch_size,-1)) for out in self.outputs]


def GPU_ABBREV(name):
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
    output_trt_model_file: str,
    onnx_model_file: str,
    min_batch_size: int,
    optimal_batch_size: int,
    max_batch_size: int,
    precision: str = "fp16",
    log_level = trt.Logger.INFO,
):
    trt_version = int(trt.__version__[:3].replace('.','')) # e.g., version 8.4.1.5 becomes 84
    gpu_name = GPU_ABBREV(torch.cuda.get_device_name())
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
    
    seq_len = network.get_input(0).shape[1]
    
    # handle dynamic shape (min/opt/max): https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes
    # by default batch dim set as 1 for all min/opt/max. If there are batch need, change the value for opt and max accordingly
    profile = trt_builder.create_optimization_profile() 
    profile.set_shape("input_ids", (min_batch_size, seq_len), (optimal_batch_size, seq_len), (max_batch_size, seq_len)) 
    profile.set_shape("attention_mask", (min_batch_size, seq_len), (optimal_batch_size, seq_len), (max_batch_size, seq_len)) 
    config.add_optimization_profile(profile)
    
    if trt_version >= 84:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4096 * (1 << 20)) # 4096 MiB, syntax after TRT 8.4
    else:
        config.max_workspace_size = 4096 * (1 << 20) # syntax before TRT 8.4

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
