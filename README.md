# tensorrt-inference

This project contains helpers to:
- quantize pytorch models with TensorRT
- Run inference with models quantied with TensorRT

It also contains notebooks to compare inference latency between vanilla Pytorch, ONNX and TensorRT inference for several models (U2Net, XLM-Roberta-base, CLIP)

## Installaton

Follow instruction here: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
1. Install CUDA >= 10.2
2. Install TensorRT via the pip wheel (https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-822/install-guide/index.html#installing-pip)
3. Install torch according to your CUDA version. Example for CUDA 11.3, `pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
4. Install torch-tensorrt integration project, `pip install torch-tensorrt -f https://github.com/NVIDIA/Torch-TensorRT/releases`
