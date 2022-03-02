import time
from contextlib import contextmanager
from typing import Dict, List, Tuple, Callable, Union

import numpy as np
import torch


def generate_random_input_for_transformers(
    n_inputs: int,
    seq_len: int,
    batch_size: int,
    include_token_ids: bool,
    device: str = "cuda"
) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, np.ndarray]]]:
    """
    Generate random inputs for Transformers

    :param n_inputs: number of inputs to generate
    :param seq_len: number of tokens
    :param batch_size: batch size
    :param include_token_ids: should we add token_type_ids
    :param device: cpu or cuda
    :return: tensors and their corresponding numpy arrays
    """
    all_inputs_pytorch = list()
    all_inputs_onnx = list()
    for _ in range(n_inputs):
        inputs_pytorch, inputs_onnx = _generate_random_input_for_transformers(
            seq_len=seq_len, batch_size=batch_size, include_token_ids=include_token_ids, device=device
        )
        all_inputs_pytorch.append(inputs_pytorch)
        all_inputs_onnx.append(inputs_onnx)
    return all_inputs_pytorch, all_inputs_onnx


def _generate_random_input_for_transformers(
    seq_len: int,
    batch_size: int,
    include_token_ids: bool,
    device: str = "cuda"
) -> Tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
    """
    Generate one random input for Transformers

    :param seq_len: number of tokens
    :param batch_size: batch size
    :param include_token_ids: should we add token_type_ids
    :param device: cpu or cuda
    :return: tensors and their corresponding numpy arrays
    """
    assert device in ["cpu", "cuda"]
    shape = (batch_size, seq_len)
    inputs_pytorch: Dict[str, torch.Tensor] = dict()
    inputs_pytorch["input_ids"] = torch.randint(high=100, size=shape, dtype=torch.long, device=device)
    if include_token_ids:
        inputs_pytorch["token_type_ids"] = torch.ones(size=shape, dtype=torch.long, device=device)
    inputs_pytorch["attention_mask"] = torch.ones(size=shape, dtype=torch.long, device=device)
    inputs_onnx: Dict[str, np.ndarray] = {
        k: np.ascontiguousarray(v.detach().cpu().numpy()) for k, v in inputs_pytorch.items()
    }
    return inputs_pytorch, inputs_onnx


def run_inference(
    inference_fn: Callable,
    inputs: List[Dict[str, Union[np.ndarray, torch.Tensor]]],
    n_measures: int
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Run inference and measure latency

    :param inference_fn: inference function
    :param inputs: model inputs (tensors for Pytorch model, numpy arrays for ONNX and TensorRT)
    :param n_measures: number of measures
    :return: model outputs and inference latencies
    """
    assert isinstance(inputs, list)
    assert len(inputs) > 0
    outputs = list()
    time_buffer: List[float] = list()
    for input in inputs:
        output = inference_fn(input)
        outputs.append(output)
    time_buffer: List[float] = list()
    for _ in range(n_measures):
        with _with_timing(time_buffer):
            _ = inference_fn(inputs[0])
    return outputs, time_buffer


def compute_mean_discrepency(
    reference_outputs: List[np.ndarray],
    test_outputs: List[np.ndarray],
    tolerance: float
) -> None:
    """
    Compare predictions. Assert differences are under a threshold.

    :param reference_outputs: reference outputs used for the comparison
    :param test_outputs: test outputs to compare against reference outputs
    :param tolerance: maximum difference between reference and test ouputs
    """
    discrepency = _compare_mean_diff(left=reference_outputs, right=test_outputs)
    assert discrepency < tolerance, (
        f"Discrepency is too big ({discrepency:.2f} > {tolerance}):\n"
        f"Reference:\n{reference_outputs}\n"
        f"VS\n"
        f"Test:\n{test_outputs}\n"
        f"Diff:\n"
        f"{np.asarray(reference_outputs) - np.asarray(test_outputs)}\n"
    )
    return discrepency


def _compare_mean_diff(left: List[np.ndarray], right: List[np.ndarray]) -> float:
    return np.mean(np.abs(np.asarray(left) - np.asarray(right)))


@contextmanager
def _with_timing(buffer: List[int]) -> None:
    """
    Measure latency

    :param buffer: placeholder for latency measures
    """
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    buffer.append(end - start)
