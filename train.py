import os
import argparse
import glob
import soundfile as sf
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import the models
from mag_CNN import MagnitudeCNN
from simplified_complex_cnn import SimplifiedComplexCNN
from dc_crn import DCCRN
from stft_pipeline import TorchSTFTPipeline

# --- HYPERPARAMETERS & UTILS ---
SR = 16000
EPS = 1e-8

def si_snr_loss(est, clean):
    """Negative SI-SNR loss"""
    def l2_norm(x):
        return torch.sum(x**2, dim=-1, keepdim=True)
    
    # zero-mean
    est = est - torch.mean(est, dim=-1, keepdim=True)
    clean = clean - torch.mean(clean, dim=-1, keepdim=True)
    
    s_c_norm = torch.sum(est * clean, dim=-1, keepdim=True)
    c_c_norm = l2_norm(clean)
    
    # target = ( <s, c> / <c, c> ) * c
    s_target = (s_c_norm / (c_c_norm + EPS)) * clean
    e_noise = est - s_target
    
    target_norm = l2_norm(s_target)
    noise_norm = l2_norm(e_noise)
    snr = 10 * torch.log10((target_norm + EPS) / (noise_norm + EPS))
    return -torch.mean(snr)


# --- DATASET ---
class WSJDataset(Dataset):
    def __init__(self, base_dir='samples/wsj', snr_levels=['0', '5', '10', '15', '20']):
        self.samples = []
        for snr in snr_levels:
            clean_dir = os.path.join(base_dir, 'clean', snr)
            noisy_dir = os.path.join(base_dir, 'noisy', snr)
            
            if not os.path.exists(clean_dir) or not os.path.exists(noisy_dir):
                continue
                
            for fname in os.listdir(clean_dir):
                if fname.endswith('.wav'):
                    clean_path = os.path.join(clean_dir, fname)
                    noisy_path = os.path.join(noisy_dir, fname)
                    if os.path.exists(noisy_path):
                        self.samples.append((noisy_path, clean_path))
        
        print(f"[{len(self.samples)} sample pairs found]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        n_path, c_path = self.samples[idx]
        
        noisy_wav, _ = sf.read(n_path)
        clean_wav, _ = sf.read(c_path)
        
        # Make them mono if needed
        if noisy_wav.ndim > 1: noisy_wav = noisy_wav.mean(axis=1)
        if clean_wav.ndim > 1: clean_wav = clean_wav.mean(axis=1)
            
        return torch.tensor(noisy_wav, dtype=torch.float32), torch.tensor(clean_wav, dtype=torch.float32)

def collate_fn(batch):
    """
    Pads all audio files in a batch to match the length of the longest
    audio file in that specific batch, so we can process full files.
    """
    noisy_list, clean_list = zip(*batch)
    
    # Find the max length in this batch
    max_len = max([x.shape[-1] for x in noisy_list])
    
    # Pad all tensors to max_len
    noisy_padded = [torch.nn.functional.pad(x, (0, max_len - x.shape[-1])) for x in noisy_list]
    clean_padded = [torch.nn.functional.pad(x, (0, max_len - x.shape[-1])) for x in clean_list]
    
    return torch.stack(noisy_padded), torch.stack(clean_padded)


# --- WRAPPERS ---
class MagCNNWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MagnitudeCNN()
        self.stft = TorchSTFTPipeline(400, 100, 512, fix=True)
        
    def forward(self, wav):
        # STFT
        spec = self.stft.stft(wav)
        real = spec[:, :257]
        imag = spec[:, 257:]
        mag = torch.sqrt(real**2 + imag**2 + EPS).unsqueeze(1) # [B, 1, F, T]
        phase = torch.atan2(imag, real)
        
        # Predict Mask (Drop DC component to get F=256 for exact layer decimation)
        mag_input = mag[:, :, 1:, :]
        enhanced_mag_in, _ = self.model(mag_input)
        
        # Re-attach the un-masked DC component
        enhanced_mag = torch.cat([mag[:, :, :1, :], enhanced_mag_in], dim=2)
        enhanced_mag = enhanced_mag.squeeze(1)
        
        # Combine with noisy phase
        est_real = enhanced_mag * torch.cos(phase)
        est_imag = enhanced_mag * torch.sin(phase)
        est_spec = torch.cat([est_real, est_imag], dim=1)
        
        # iSTFT
        out_wav = self.stft.istft(est_spec).squeeze(1)
        return torch.clamp(out_wav, -1, 1)


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all', choices=['mag_cnn', 'dcunet', 'dccrn', 'all'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps for large effective batches')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.accumulation_steps > 1:
        print(f"Using gradient accumulation! Effective batch size: {args.batch_size * args.accumulation_steps}")

    # Dataset / DataLoader
    print("Loading datasets...")
    dataset = WSJDataset()
    if len(dataset) == 0:
        print("No valid data found in samples/wsj/. Exiting.")
        return
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)
    
    models_to_train = ['mag_cnn', 'dcunet', 'dccrn'] if args.model == 'all' else [args.model]

    for model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"       TRAINING MODEL: {model_name.upper()}")
        print(f"{'='*50}")

        # 1. Instantiate the Model
        if model_name == 'mag_cnn':
            model = MagCNNWrapper()
        elif model_name == 'dcunet':
            model = SimplifiedComplexCNN()
        elif model_name == 'dccrn':
            model = DCCRN(rnn_units=256, masking_mode='E', use_clstm=True, kernel_num=[32, 64, 128, 256, 256, 256])

        model = model.to(device)
        
        # 2. Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # 3. Training Loop
        os.makedirs(args.save_dir, exist_ok=True)
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            
            optimizer.zero_grad() # Initialize gradients outside the loop for accumulation
            
            for batch_idx, (noisy, clean) in enumerate(dataloader):
                noisy, clean = noisy.to(device), clean.to(device)
                
                # Forward pass
                if model_name == 'mag_cnn':
                    est_wav = model(noisy)
                elif model_name == 'dcunet':
                    _, est_wav = model(noisy)
                elif model_name == 'dccrn':
                    est_wav = model(noisy)[1]
                    if isinstance(est_wav, list):
                        est_wav = est_wav[-1]
                
                # Match lengths (in case iSTFT padding is off)
                min_len = min(clean.shape[-1], est_wav.shape[-1])
                est_wav = est_wav[..., :min_len]
                clean = clean[..., :min_len]
                
                # Calculate SI-SNR
                loss = si_snr_loss(est_wav, clean)
                
                # Scale loss by accumulation steps
                scaled_loss = loss / args.accumulation_steps
                scaled_loss.backward()
                
                # Update weights only after enough steps
                if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item()
                
                print(f"  Epoch [{epoch}/{args.epochs}] Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}", end='\r')
                
            
            if len(dataloader) > 0:
                avg_loss = epoch_loss / len(dataloader)
            else:
                avg_loss = 0.0
            print(f"\n=== {model_name.upper()} | Epoch {epoch} Summary | Avg Loss: {avg_loss:.4f} ===")
            
            # Save Checkpoint
            save_path = os.path.join(args.save_dir, f"{model_name}_epoch_{epoch}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")


if __name__ == '__main__':
    main()
