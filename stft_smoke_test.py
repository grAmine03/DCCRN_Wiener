"""Smoke tests for shared STFT pipelines.

This script checks analysis->synthesis round-trip error for:
1) NumPy STFTPipeline (used by Wiener flow)
2) TorchSTFTPipeline (used by DCCRN flow)
"""

import argparse

import numpy as np
import torch

from stft_pipeline import STFTConfig, STFTPipeline, TorchSTFTPipeline


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def run_numpy_roundtrip(length: int, frame_length: int, hop_length: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(length).astype(np.float64) * 0.05

    pipeline = STFTPipeline(STFTConfig(frame_length=frame_length, hop_length=hop_length, window="hann"))
    spec, original_len = pipeline.stft(signal)
    recon = pipeline.istft(spec, output_length=original_len)

    return mse(signal[: len(recon)], recon)


def run_torch_roundtrip(length: int, frame_length: int, hop_length: int, fft_length: int, seed: int) -> float:
    torch.manual_seed(seed)
    waveform = torch.randn(1, length, dtype=torch.float32) * 0.05

    pipeline = TorchSTFTPipeline(
        win_len=frame_length,
        win_inc=hop_length,
        fft_len=fft_length,
        win_type="hann",
        fix=True,
    )

    spec = pipeline.stft(waveform)
    recon = pipeline.istft(spec).squeeze(1)

    common_len = min(waveform.shape[-1], recon.shape[-1])
    ref = waveform[:, :common_len].detach().cpu().numpy()
    out = recon[:, :common_len].detach().cpu().numpy()
    return mse(ref, out)


def parse_args():
    parser = argparse.ArgumentParser(description="Run STFT round-trip smoke tests.")
    parser.add_argument("--length", type=int, default=32000, help="Waveform length in samples.")
    parser.add_argument("--frame-length", type=int, default=512, help="STFT frame length.")
    parser.add_argument("--hop-length", type=int, default=256, help="STFT hop length.")
    parser.add_argument("--fft-length", type=int, default=512, help="FFT length for torch conv STFT.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()

    numpy_mse = run_numpy_roundtrip(
        length=args.length,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        seed=args.seed,
    )
    torch_mse = run_torch_roundtrip(
        length=args.length,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        fft_length=args.fft_length,
        seed=args.seed,
    )

    print("STFT Smoke Test Results")
    print(f"- NumPy pipeline (Wiener path) MSE: {numpy_mse:.12e}")
    print(f"- Torch pipeline (DCCRN path) MSE: {torch_mse:.12e}")


if __name__ == "__main__":
    main()
