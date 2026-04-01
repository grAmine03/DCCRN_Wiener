import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.time_pad = padding[1]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, (padding[0], 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        if self.time_pad > 0:
            x = F.pad(x, [self.time_pad, 0, 0, 0])
        return self.prelu(self.bn(self.conv(x)))

class TrConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(TrConvBlock, self).__init__()
        self.tr_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.bn(self.tr_conv(x)))

class MagnitudeCNN(nn.Module):
    def __init__(self, win_len=400, win_inc=100, fft_len=512, kernel_num=[16, 32, 64, 128, 256, 256]):
        super(MagnitudeCNN, self).__init__()
        
        self.fft_len = fft_len
        self.win_len = win_len
        self.win_inc = win_inc
        # Assumes input is magnitude spectrogram: [B, 1, F, T]
        
        self.kernel_num = [1] + kernel_num
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        # Encoder
        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                ConvBlock(
                    self.kernel_num[idx], 
                    self.kernel_num[idx+1], 
                    kernel_size=(5, 2), 
                    stride=(2, 1), 
                    padding=(2, 1) # Note: Custom causal padding might be needed for T dimension
                )
            )
            
        # Decoder
        for idx in range(len(self.kernel_num) - 1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    TrConvBlock(
                        self.kernel_num[idx] * 2, # *2 for skip connections
                        self.kernel_num[idx-1],
                        kernel_size=(5, 2),
                        stride=(2, 1),
                        padding=(2, 0),
                        output_padding=(1, 0)
                    )
                )
            else:
                # Last layer without BN and PReLU, outputting mask
                self.decoder.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx-1],
                            kernel_size=(5, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        ),
                        nn.Sigmoid() # Mask between 0 and 1
                    )
                )

    def forward(self, mag_spec):
        # mag_spec: [B, 1, F, T]
        
        out = mag_spec
        encoder_outs = []
        
        for layer in self.encoder:
            out = layer(out)
            encoder_outs.append(out)
            
        # Bottleneck could go here (e.g. LSTM), but sticking to pure CNN
        
        for idx in range(len(self.decoder)):
            # Skip connection
            out = torch.cat([out, encoder_outs[-1 - idx]], dim=1)
            out = self.decoder[idx](out)
            out = out[..., 1:] # Remove rightmost padded frame to keep time dimension consistent
            
        mask = out # [B, 1, F, T]
        
        # Apply mask
        enhanced_mag = mag_spec * mask
        
        return enhanced_mag, mask

