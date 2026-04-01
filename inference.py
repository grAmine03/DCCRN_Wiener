import os
import argparse
import soundfile as sf
import torch

from mag_CNN import MagnitudeCNN
from simplified_complex_cnn import SimplifiedComplexCNN
from dc_crn import DCCRN
from train import MagCNNWrapper

def main():
    parser = argparse.ArgumentParser(description="Generate enhanced wav files for compare_methods.py")
    parser.add_argument('--model', type=str, default='all', choices=['mag_cnn', 'dcunet', 'dccrn', 'all'], help='Model architecture to use')
    parser.add_argument('--epoch', type=int, default=10, help='Epoch checkpoint to load (e.g. 10 for epoch_10.pt)')
    parser.add_argument('--input_dir', type=str, default='samples/wsj/noisy', help='Path to the noisy dataset')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    models_to_infer = ['mag_cnn', 'dcunet', 'dccrn'] if args.model == 'all' else [args.model]

    for model_name in models_to_infer:
        print(f"\n{'='*50}")
        print(f"       INFERENCE FOR: {model_name.upper()}")
        print(f"{'='*50}")

        # 1. Load Model Architecture
        if model_name == 'mag_cnn':
            model = MagCNNWrapper()
        elif model_name == 'dcunet':
            model = SimplifiedComplexCNN()
        elif model_name == 'dccrn':
            model = DCCRN(rnn_units=256, masking_mode='E', use_clstm=True, kernel_num=[32, 64, 128, 256, 256, 256])
        
        # 2. Checkpoint & Output paths
        checkpoint_path = f"checkpoints/{model_name}_epoch_{args.epoch}.pt"
        output_dir = f"samples/wsj/{model_name}_trained"

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}\nSkipping {model_name}.")
            continue

        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()

        # 3. Process the Data
        snr_levels = ['0', '5', '10', '15', '20']
        
        with torch.no_grad():
            for snr in snr_levels:
                in_snr_dir = os.path.join(args.input_dir, snr)
                out_snr_dir = os.path.join(output_dir, snr)
                
                if not os.path.exists(in_snr_dir):
                    continue
                    
                os.makedirs(out_snr_dir, exist_ok=True)
                files = [f for f in os.listdir(in_snr_dir) if f.endswith('.wav')]
                
                for fname in files:
                    in_path = os.path.join(in_snr_dir, fname)
                    out_path = os.path.join(out_snr_dir, fname)
                    
                    # Load Audio
                    noisy_wav, sr = sf.read(in_path)
                    noisy_tensor = torch.tensor(noisy_wav, dtype=torch.float32).unsqueeze(0).to(device) # [1, T]
                    
                    # Forward Pass
                    if model_name == 'mag_cnn':
                        est_wav = model(noisy_tensor)
                    elif model_name == 'dcunet':
                        _, est_wav = model(noisy_tensor)
                    elif model_name == 'dccrn':
                        est_wav = model(noisy_tensor)[1]
                        if isinstance(est_wav, list):
                            est_wav = est_wav[-1]
                            
                    # Ensure the shape is 1D and matched to original length
                    est_wav_np = est_wav.squeeze().cpu().numpy()
                    est_wav_np = est_wav_np[:len(noisy_wav)]
                    
                    # Save Enhanced Audio
                    sf.write(out_path, est_wav_np, sr)
                
                print(f"Processed SNR {snr}: saved {len(files)} files to {out_snr_dir}")

    print("\nInference complete! You can now uncomment them in compare_methods.py and run it.")

if __name__ == '__main__':
    main()