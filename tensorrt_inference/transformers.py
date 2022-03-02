import numpy as np
import tensorrt as trt
import pycuda.driver as cuda


class MinMaxCalibratorTransformers(trt.IInt8MinMaxCalibrator):
    """
    Calibrator used for Transformers

    Calibration is required for INT8 quantization. Its purpose is to feed
    examples to a trained model to compute the dynamic range of tensors in
    order to know what are the important values to represent, and thus how to choose 
    the scale used to convert a quantized tensor to an unquantized tensor

    Min/max calibrator are advised for BERT-base models
    (https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#enable_int8_c)
    """

    def __init__(self, examples):
        trt.IInt8MinMaxCalibrator.__init__(self)

        self.current_index = 0
        self.examples = examples
        self.n_examples = len(examples)
        self.batch_size = examples[0]["input_ids"].shape[0]
        
        nbytes_input_ids = examples[0]["input_ids"].nbytes
        self.device_input_ids = cuda.mem_alloc(nbytes_input_ids)
        
        nbytes_attention_mask = examples[0]["attention_mask"].nbytes
        self.device_attention_mask = cuda.mem_alloc(nbytes_attention_mask)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= self.n_examples:
            return None

        input_ids = np.ascontiguousarray(
            self.examples[self.current_index]["input_ids"].ravel()
        )
        cuda.memcpy_htod(self.device_input_ids, input_ids)
        
        attention_mask = np.ascontiguousarray(
            self.examples[self.current_index]["attention_mask"].ravel()
        )
        cuda.memcpy_htod(self.device_attention_mask, attention_mask)

        self.current_index += 1
        return [self.device_input_ids, self.device_attention_mask]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        return None
