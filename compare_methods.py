import csv
import os

import librosa
import numpy as np
import soundfile as sf

try:
    from pesq import pesq as pesq_fn
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False


EPS = 1e-8
N_FFT = 512
HOP_LENGTH = 256
METRICS = ["si_snr", "mpe", "wmpe", "mae_deg", "plv", "p15", "p30", "pesq"]


def load_audio(path):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    return wav.astype(np.float32), sr


def si_snr(est, ref, eps=EPS):
    est = est - np.mean(est)
    ref = ref - np.mean(ref)
    alpha = np.sum(est * ref) / (np.sum(ref ** 2) + eps)
    target = alpha * ref
    noise = est - target
    return 10.0 * np.log10((np.sum(target ** 2) / (np.sum(noise ** 2) + eps)) + eps)


def safe_pesq(sr, ref, est):
    if not PESQ_AVAILABLE:
        return np.nan
    if sr not in (8000, 16000):
        return np.nan

    mode = "nb" if sr == 8000 else "wb"
    ref = np.clip(ref, -1.0, 1.0).astype(np.float32)
    est = np.clip(est, -1.0, 1.0).astype(np.float32)

    try:
        return float(pesq_fn(sr, ref, est, mode))
    except Exception:
        return np.nan


def compute_phase_metrics(ref, est):
    ref_stft = librosa.stft(ref, n_fft=N_FFT, hop_length=HOP_LENGTH)
    est_stft = librosa.stft(est, n_fft=N_FFT, hop_length=HOP_LENGTH)

    n_freq = min(ref_stft.shape[0], est_stft.shape[0])
    n_frames = min(ref_stft.shape[1], est_stft.shape[1])
    ref_stft = ref_stft[:n_freq, :n_frames]
    est_stft = est_stft[:n_freq, :n_frames]

    dphi = np.angle(np.exp(1j * (np.angle(est_stft) - np.angle(ref_stft))))
    phase_error = 1.0 - np.cos(dphi)

    w = np.abs(ref_stft)
    metrics = {
        "mpe": float(np.mean(phase_error)),
        "wmpe": float(np.sum(w * phase_error) / (np.sum(w) + EPS)),
        "mae_deg": float(np.mean(np.abs(dphi)) * 180.0 / np.pi),
        "plv": float(np.abs(np.mean(np.exp(1j * dphi)))),
        "p15": float(np.mean(np.abs(dphi) < np.deg2rad(15.0))),
        "p30": float(np.mean(np.abs(dphi) < np.deg2rad(30.0))),
    }
    return metrics


def evaluate_pair(ref_wav, est_wav, sr):
    min_len = min(len(ref_wav), len(est_wav))
    ref = ref_wav[:min_len]
    est = est_wav[:min_len]

    result = {
        "si_snr": float(si_snr(est, ref)),
        "pesq": float(safe_pesq(sr, ref, est)),
    }
    result.update(compute_phase_metrics(ref, est))
    return result


def init_method_bucket():
    return {metric: [] for metric in METRICS}


def add_metrics(bucket, metrics):
    for metric in METRICS:
        value = metrics.get(metric, np.nan)
        if not np.isnan(value):
            bucket[metric].append(float(value))


def aggregate_means(bucket):
    return {
        metric: (float(np.mean(values)) if len(values) > 0 else np.nan)
        for metric, values in bucket.items()
    }


def format_num(value, precision=3, as_percent=False):
    if np.isnan(value):
        return "n/a"
    if as_percent:
        return f"{100.0 * value:.1f}%"
    return f"{value:.{precision}f}"


def print_summary(title, method_metrics):
    print(f"\n=== {title} ===")
    ranked = sorted(
        method_metrics.items(),
        key=lambda item: item[1]["si_snr"] if not np.isnan(item[1]["si_snr"]) else -1e9,
        reverse=True,
    )

    for rank, (method, stats) in enumerate(ranked, start=1):
        line = (
            f"{rank:2d}. {method:22} "
            f"SI-SNR={format_num(stats['si_snr'], 2)} dB  "
            f"PESQ={format_num(stats['pesq'], 3)}  "
            f"wMPE={format_num(stats['wmpe'], 4)}  "
            f"PLV={format_num(stats['plv'], 4)}  "
            f"<30deg={format_num(stats['p30'], as_percent=True)}"
        )
        print(line)


def main():
    base_dir = "samples/wsj"
    clean_root = os.path.join(base_dir, "clean")

    methods = {
        "Noisy": "noisy",
        "Wiener Filter": "wiener",
        "Mag-only CNN (CRN)": "crn",
        "Complex CNN (DCUNET)": "dcunet",
        "DCCRN": "dccrn_e",
        "Trained Mag-CNN": "mag_cnn_trained",
        "Trained DCUNET": "dcunet_trained",
        "Trained DCCRN": "dccrn_trained",
    }

    if not os.path.isdir(clean_root):
        print(f"Missing clean directory: {clean_root}")
        return

    snr_levels = [
        d
        for d in sorted(os.listdir(clean_root), key=lambda x: float(x))
        if os.path.isdir(os.path.join(clean_root, d))
    ]
    if not snr_levels:
        print("No SNR folders found under samples/wsj/clean")
        return

    overall = {method: init_method_bucket() for method in methods}
    per_snr = {snr: {method: init_method_bucket() for method in methods} for snr in snr_levels}
    rows = []

    total_pairs = 0
    for snr in snr_levels:
        clean_dir = os.path.join(clean_root, snr)
        files = sorted([f for f in os.listdir(clean_dir) if f.lower().endswith(".wav")])
        if not files:
            print(f"No wav files in {clean_dir}")
            continue

        for filename in files:
            clean_path = os.path.join(clean_dir, filename)
            ref_wav, ref_sr = load_audio(clean_path)

            for method_name, method_folder in methods.items():
                est_path = os.path.join(base_dir, method_folder, snr, filename)
                if not os.path.exists(est_path):
                    continue

                est_wav, est_sr = load_audio(est_path)
                if est_sr != ref_sr:
                    est_wav = librosa.resample(est_wav, orig_sr=est_sr, target_sr=ref_sr)

                metrics = evaluate_pair(ref_wav, est_wav, ref_sr)
                add_metrics(overall[method_name], metrics)
                add_metrics(per_snr[snr][method_name], metrics)

                row = {"snr": snr, "file": filename, "method": method_name}
                row.update(metrics)
                rows.append(row)
                total_pairs += 1

    print(f"Evaluated method/file pairs: {total_pairs}")
    if not PESQ_AVAILABLE:
        print("PESQ package not found. Install with: pip install pesq")

    overall_means = {method: aggregate_means(bucket) for method, bucket in overall.items()}
    print_summary("Overall Mean Metrics", overall_means)

    for snr in snr_levels:
        snr_means = {method: aggregate_means(bucket) for method, bucket in per_snr[snr].items()}
        print_summary(f"SNR = {snr} dB", snr_means)

    out_csv = "evaluation_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["snr", "file", "method"] + METRICS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved per-file metrics to {out_csv}")


if __name__ == "__main__":
    main()