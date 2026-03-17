"""Denoise speech with a sliding-window Wiener filter.

Noise power is estimated from an initial silence segment of the noisy recording.

Example:
    python wiener_filter.py \
        --input noisy_pink_0dB.wav \
        --output enhanced.wav \
        --noise-duration 0.25
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


EPS = 1e-12


def load_audio_mono(path: Path):
    waveform, sample_rate = sf.read(str(path), always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=1)
    return waveform, sample_rate


def frame_signal(signal: np.ndarray, frame_length: int, hop_length: int):
    if len(signal) == 0:
        raise ValueError("Input signal is empty.")

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


def overlap_add(frames: np.ndarray, frame_length: int, hop_length: int, output_len: int, window: np.ndarray):
    num_frames = frames.shape[0]
    total_len = (num_frames - 1) * hop_length + frame_length
    output = np.zeros(total_len, dtype=np.float64)
    weight = np.zeros(total_len, dtype=np.float64)

    window_sq = window ** 2
    for i in range(num_frames):
        start = i * hop_length
        output[start : start + frame_length] += frames[i] * window
        weight[start : start + frame_length] += window_sq

    output /= (weight + EPS)
    return output[:output_len]


def estimate_noise_psd(noisy_spec: np.ndarray, sample_rate: int, hop_length: int, noise_duration: float):
    noise_samples = int(max(noise_duration, 0.0) * sample_rate)
    frame_starts = np.arange(noisy_spec.shape[0]) * hop_length
    noise_mask = frame_starts < noise_samples

    if not np.any(noise_mask):
        noise_mask[0] = True

    noise_psd = np.mean(np.abs(noisy_spec[noise_mask]) ** 2, axis=0)
    return noise_psd + EPS


def wiener_filter_sliding(
    noisy: np.ndarray,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
    noise_duration: float,
    alpha: float,
):
    window = np.hanning(frame_length)

    frames, original_len = frame_signal(noisy, frame_length, hop_length)
    windowed = frames * window[None, :]
    noisy_spec = np.fft.rfft(windowed, axis=1)

    noise_psd = estimate_noise_psd(noisy_spec, sample_rate, hop_length, noise_duration)

    enhanced_spec = np.zeros_like(noisy_spec)
    prev_clean_psd = np.zeros_like(noise_psd)

    for i in range(noisy_spec.shape[0]):
        yk = noisy_spec[i]
        post_snr = (np.abs(yk) ** 2) / noise_psd
        inst_prior = np.maximum(post_snr - 1.0, 0.0)
        prior_snr = alpha * (prev_clean_psd / noise_psd) + (1.0 - alpha) * inst_prior

        gain = prior_snr / (1.0 + prior_snr)
        xk = gain * yk

        enhanced_spec[i] = xk
        prev_clean_psd = np.abs(xk) ** 2

    enhanced_frames = np.fft.irfft(enhanced_spec, n=frame_length, axis=1)
    enhanced = overlap_add(enhanced_frames, frame_length, hop_length, original_len, window)
    return enhanced


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply a sliding-window Wiener filter with noise estimated from initial silence."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to noisy input audio file.")
    parser.add_argument("--output", type=Path, required=True, help="Path to save denoised audio file.")
    parser.add_argument(
        "--noise-duration",
        type=float,
        default=0.25,
        help="Initial silence duration (seconds) used to estimate stationary noise.",
    )
    parser.add_argument("--frame-length", type=int, default=512, help="STFT frame length in samples.")
    parser.add_argument("--hop-length", type=int, default=256, help="Frame hop length in samples.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.98,
        help="Decision-directed smoothing factor in [0, 1). Higher gives smoother gain.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.frame_length <= 0:
        raise ValueError("--frame-length must be positive.")
    if args.hop_length <= 0:
        raise ValueError("--hop-length must be positive.")
    if args.hop_length > args.frame_length:
        raise ValueError("--hop-length must be <= --frame-length.")
    if not (0.0 <= args.alpha < 1.0):
        raise ValueError("--alpha must satisfy 0 <= alpha < 1.")

    noisy, sample_rate = load_audio_mono(args.input)
    enhanced = wiener_filter_sliding(
        noisy=noisy,
        sample_rate=sample_rate,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        noise_duration=args.noise_duration,
        alpha=args.alpha,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), enhanced.astype(np.float32), sample_rate, subtype="FLOAT")
    print(f"Saved denoised file: {args.output}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Noise estimation duration: {args.noise_duration:.3f} s")
    print(f"Frame length / hop: {args.frame_length} / {args.hop_length}")


if __name__ == "__main__":
    main()
