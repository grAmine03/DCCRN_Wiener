import torch
import torch.nn as nn
import torch.nn.functional as F

from complexnn import ComplexConv2d, ComplexConvTranspose2d, ComplexBatchNorm, complex_cat
from stft_pipeline import TorchSTFTPipeline

class ComplexConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1), use_cbn=True):
        super().__init__()
        self.conv = ComplexConv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.cbn = ComplexBatchNorm(out_channels) if use_cbn else nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.cbn(self.conv(x)))

class TrComplexConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0), output_padding=(1, 0), use_cbn=True, is_last=False):
        super().__init__()
        self.tr_conv = ComplexConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.is_last = is_last
        if not is_last:
            self.cbn = ComplexBatchNorm(out_channels) if use_cbn else nn.BatchNorm2d(out_channels)
            self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.tr_conv(x)
        if not self.is_last:
            x = self.prelu(self.cbn(x))
        return x

class SimplifiedComplexCNN(nn.Module):
    """
    Simplified Complex-valued CNN for speech denoising.
    Uses Complex Convolutions and Skip Connections (U-Net style).
    No LSTM bottlenecks to keep it purely convolutional and simple.
    """
    def __init__(self, win_len=400, win_inc=100, fft_len=512, kernel_num=[72, 72, 144, 144, 144, 160, 160, 180]):
        super().__init__()
        
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        
        self.stft_pipeline = TorchSTFTPipeline(win_len=win_len, win_inc=win_inc, fft_len=fft_len, fix=True)
        
        self.kernel_num = [2] + kernel_num # 2 channels for input (Real, Imag)
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        # Encoder
        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                ComplexConvBlock(
                    self.kernel_num[idx], 
                    self.kernel_num[idx+1], 
                )
            )
            
        # Decoder
        for idx in range(len(self.kernel_num) - 1, 0, -1):
            is_last = (idx == 1)
            self.decoder.append(
                TrComplexConvBlock(
                    self.kernel_num[idx] * 2, # *2 for skip connections
                    self.kernel_num[idx-1],
                    is_last=is_last
                )
            )

    def forward(self, inputs):
        # 1. STFT
        specs = self.stft_pipeline.stft(inputs) # [B, F_complex, T]
        real = specs[:, :self.fft_len//2+1]
        imag = specs[:, self.fft_len//2+1:]
        
        # [B, 2, F, T]
        cspecs = torch.stack([real, imag], dim=1)
        cspecs = cspecs[:, :, 1:] # Remove DC component for network input
        
        out = cspecs
        encoder_outs = []
        
        # 2. Encode
        for layer in self.encoder:
            out = layer(out)
            encoder_outs.append(out)
            
        # 3. Decode with Skip Connections
        for idx in range(len(self.decoder)):
            # complex_cat ensures Real-Real and Imag-Imag concatenation
            out = complex_cat([out, encoder_outs[-1 - idx]], axis=1)
            out = self.decoder[idx](out)
            out = out[..., 1:] # Padding correction to match time steps
            
        # out is Complex Mask: [B, 2, F-1, T]
        mask_real = out[:, 0]
        mask_imag = out[:, 1]
        
        # Pad back the DC component
        mask_real = F.pad(mask_real, [0, 0, 1, 0])
        mask_imag = F.pad(mask_imag, [0, 0, 1, 0])
        
        # 4. Apply Complex Masking (Masking Mode C)
        est_real = real * mask_real - imag * mask_imag
        est_imag = real * mask_imag + imag * mask_real
        
        # 5. iSTFT
        out_spec = torch.cat([est_real, est_imag], dim=1)
        out_wav = self.stft_pipeline.istft(out_spec)
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp_(out_wav, -1, 1)
        
        return out_spec, out_wav

def si_snr(s1, s2, eps=1e-8):
    def l2_norm(x):
        return torch.sum(x**2, dim=-1, keepdim=True)
    
    s1_s2_norm = torch.sum(s1 * s2, dim=-1, keepdim=True)
    s2_s2_norm = l2_norm(s2)
    s_target = (s1_s2_norm / (s2_s2_norm + eps)) * s2
    e_noise = s1 - s_target
    
    target_norm = l2_norm(s_target)
    noise_norm = l2_norm(e_noise)
    snr = 10 * torch.log10((target_norm + eps) / (noise_norm + eps))
    return torch.mean(snr)

def train_dummy():
    print("Initializing Simplified Complex CNN...")
    model = SimplifiedComplexCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Generating dummy data...")
    # [Batch, Time] -> e.g., 2 seconds of audio at 16kHz
    noisy_wavs = torch.randn(4, 16000 * 2).clamp_(-1, 1)
    clean_wavs = torch.randn(4, 16000 * 2).clamp_(-1, 1)
    
    print("Starting Dummy Training Step...")
    model.train()
    optimizer.zero_grad()
    
    out_spec, est_wavs = model(noisy_wavs)
    
    # Negative SI-SNR loss
    loss = -si_snr(est_wavs, clean_wavs)
    
    loss.backward()
    optimizer.step()
    
    print(f"Training step complete. Loss: {loss.item():.4f}")

if __name__ == '__main__':
    train_dummy()
