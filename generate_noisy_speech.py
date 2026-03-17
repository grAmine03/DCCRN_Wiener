"""Generate noisy speech by mixing clean speech and stationary noise at a target SNR.

Examples:
    python generate_noisy_speech.py \
        --clean samples/wsj/clean/0/example.wav \
        --output mixed.wav \
        --noise-type white \
        --snr-db 5

    python generate_noisy_speech.py \
        --clean clean.wav \
        --noise-path noise.wav \
        --snr-db 0 \
        --output noisy.wav
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


EPS = 1e-12


def load_audio_mono(path: Path):
    """Load audio file and return mono float waveform plus sample rate."""
    waveform, sample_rate = sf.read(str(path), always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=1)
    return waveform, sample_rate


def match_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Repeat or trim signal to exactly target_length samples."""
    if len(signal) == target_length:
        return signal
    if len(signal) > target_length:
        return signal[:target_length]

    repeats = int(np.ceil(target_length / len(signal)))
    tiled = np.tile(signal, repeats)
    return tiled[:target_length]


def generate_white_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(length)


def generate_pink_noise(length: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Generate approximate pink (1/f) noise using frequency-domain shaping."""
    white = rng.standard_normal(length)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length, d=1.0 / sample_rate)

    scale = np.ones_like(freqs)
    scale[0] = 0.0
    nonzero = freqs > 0
    scale[nonzero] = 1.0 / np.sqrt(freqs[nonzero])

    pink = np.fft.irfft(spectrum * scale, n=length)
    pink_std = np.std(pink)
    if pink_std < EPS:
        return pink
    return pink / pink_std


def power(signal: np.ndarray) -> float:
    return float(np.mean(signal ** 2))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float):
    """Scale noise so that clean/noise power ratio equals target SNR (dB)."""
    clean_power = power(clean)
    noise_power = power(noise)
    if clean_power < EPS:
        raise ValueError("Clean signal power is zero; cannot set SNR.")
    if noise_power < EPS:
        raise ValueError("Noise signal power is zero; cannot set SNR.")

    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    scaled_noise = noise * scale
    mixed = clean + scaled_noise
    return mixed, scaled_noise, scale


def parse_args():
    parser = argparse.ArgumentParser(description="Generate noisy speech at a target SNR.")
    parser.add_argument("--clean", type=Path, required=True, help="Path to clean speech WAV/FLAC file.")
    parser.add_argument("--output", type=Path, required=True, help="Path to save mixed noisy speech.")
    parser.add_argument("--snr-db", type=float, required=True, help="Target SNR in dB.")

    parser.add_argument(
        "--noise-type",
        choices=["white", "pink"],
        default="white",
        help="Stationary noise type to generate (ignored if --noise-path is given).",
    )
    parser.add_argument(
        "--noise-path",
        type=Path,
        default=None,
        help="Optional path to external noise file. If provided, it is used instead of generated noise.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible noise generation.")
    return parser.parse_args()


def main():
    args = parse_args()

    clean, sample_rate = load_audio_mono(args.clean)
    if len(clean) == 0:
        raise ValueError("Clean signal is empty.")

    rng = np.random.default_rng(args.seed)
    if args.noise_path is not None:
        noise, noise_sr = load_audio_mono(args.noise_path)
        if noise_sr != sample_rate:
            raise ValueError(
                f"Sample rate mismatch: clean={sample_rate} Hz, noise={noise_sr} Hz. "
                "Please resample one of the files so they match."
            )
        noise = match_length(noise, len(clean))
    else:
        if args.noise_type == "white":
            noise = generate_white_noise(len(clean), rng)
        else:
            noise = generate_pink_noise(len(clean), sample_rate, rng)

    mixed, scaled_noise, applied_scale = mix_at_snr(clean, noise, args.snr_db)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), mixed.astype(np.float32), sample_rate, subtype="FLOAT")

    achieved_snr = 10.0 * np.log10(power(clean) / (power(scaled_noise) + EPS))
    print(f"Saved noisy file: {args.output}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Target SNR: {args.snr_db:.2f} dB")
    print(f"Achieved SNR: {achieved_snr:.2f} dB")
    print(f"Applied noise scale factor: {applied_scale:.6f}")


if __name__ == "__main__":
    main()
