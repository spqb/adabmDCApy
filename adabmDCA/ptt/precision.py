"""PTT accumulation precision: float32 on MPS, float64 elsewhere.

Small diagnostic vectors may be transferred to CPU for float64 reductions;
model tensors and persistent per-chain state remain on their original device.
"""

import torch


def accumulation_dtype(device):
    return torch.float32 if torch.device(device).type == "mps" else torch.float64


def device_accumulation(value):
    return value.to(dtype=accumulation_dtype(value.device))


def diagnostic_double(value):
    return value.cpu().double() if value.device.type == "mps" else value.double()
