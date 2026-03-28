"""Reusable STFT pipeline utilities for analysis and overlap-add reconstruction.

This module provides a small, explicit pipeline:
1) analysis: waveform -> complex STFT frames
2) processing: user-defined operation in STFT domain
3) synthesis: complex STFT frames -> waveform
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from conv_stft import ConvSTFT, ConviSTFT


EPS = 1e-12


@dataclass
class STFTConfig:
    frame_length: int = 512
    hop_length: int = 256
    window: str = "hann"


class STFTPipeline:
    """Short-time Fourier transform pipeline with perfect-reconstruction overlap-add."""

    def __init__(self, config: STFTConfig):
        if config.frame_length <= 0:
            raise ValueError("frame_length must be positive.")
        if config.hop_length <= 0:
            raise ValueError("hop_length must be positive.")
        if config.hop_length > config.frame_length:
            raise ValueError("hop_length must be <= frame_length.")
        if config.window != "hann":
            raise ValueError("Only 'hann' window is supported currently.")

        self.config = config
        self.window = np.hanning(config.frame_length).astype(np.float64)

    def _frame_signal(self, signal: np.ndarray):
        if signal.ndim != 1:
            raise ValueError("signal must be a 1-D mono waveform.")
        if len(signal) == 0:
            raise ValueError("signal must not be empty.")

        frame_length = self.config.frame_length
        hop_length = self.config.hop_length

        if len(signal) <= frame_length:
            num_frames = 1
        else:
            num_frames = 1 + int(np.ceil((len(signal) - frame_length) / hop_length))

        padded_len = (num_frames - 1) * hop_length + frame_length
        pad_amount = padded_len - len(signal)
        padded = np.pad(signal, (0, pad_amount), mode="constant")

        frames = np.zeros((num_frames, frame_length), dtype=np.float64)
        for i in range(num_frames):
            start = i * hop_length
            frames[i] = padded[start : start + frame_length]

        return frames, len(signal)

    def stft(self, signal: np.ndarray):
        """Compute STFT and return (complex_spectrogram, original_signal_length)."""
        frames, original_length = self._frame_signal(signal)
        windowed = frames * self.window[None, :]
        spectrogram = np.fft.rfft(windowed, axis=1)
        return spectrogram, original_length

    def istft(self, spectrogram: np.ndarray, output_length: int):
        """Reconstruct waveform from STFT with weighted overlap-add."""
        if spectrogram.ndim != 2:
            raise ValueError("spectrogram must be [num_frames, num_bins].")
        if output_length <= 0:
            raise ValueError("output_length must be positive.")

        frame_length = self.config.frame_length
        hop_length = self.config.hop_length

        frames = np.fft.irfft(spectrogram, n=frame_length, axis=1)
        num_frames = frames.shape[0]

        total_len = (num_frames - 1) * hop_length + frame_length
        output = np.zeros(total_len, dtype=np.float64)
        weight = np.zeros(total_len, dtype=np.float64)

        window_sq = self.window ** 2
        for i in range(num_frames):
            start = i * hop_length
            output[start : start + frame_length] += frames[i] * self.window
            weight[start : start + frame_length] += window_sq

        output /= (weight + EPS)
        return output[:output_length]

    def run(self, signal: np.ndarray, process_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        """Run analysis -> optional spectral processing -> synthesis."""
        spectrogram, original_length = self.stft(signal)
        processed = process_fn(spectrogram) if process_fn is not None else spectrogram
        return self.istft(processed, output_length=original_length)


class TorchSTFTPipeline(nn.Module):
    """Differentiable STFT pipeline backed by convolutional analysis/synthesis kernels."""

    def __init__(
        self,
        win_len: int,
        win_inc: int,
        fft_len: int,
        win_type: str = "hann",
        fix: bool = True,
    ):
        super().__init__()
        self.stft_module = ConvSTFT(win_len, win_inc, fft_len, win_type, "complex", fix=fix)
        self.istft_module = ConviSTFT(win_len, win_inc, fft_len, win_type, "complex", fix=fix)

    def stft(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.stft_module(waveform)

    def istft(self, complex_spectrogram: torch.Tensor) -> torch.Tensor:
        return self.istft_module(complex_spectrogram)

    def run(
        self,
        waveform: torch.Tensor,
        process_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        specs = self.stft(waveform)
        processed = process_fn(specs) if process_fn is not None else specs
        return self.istft(processed)
