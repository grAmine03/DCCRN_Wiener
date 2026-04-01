import os
import subprocess

base_dir = 'samples/wsj'
noisy_root = os.path.join(base_dir, 'noisy')
wiener_root = os.path.join(base_dir, 'wiener')

for snr in os.listdir(noisy_root):
    noisy_dir = os.path.join(noisy_root, snr)
    if not os.path.isdir(noisy_dir): continue
    
    wiener_dir = os.path.join(wiener_root, snr)
    os.makedirs(wiener_dir, exist_ok=True)
    
    for filename in os.listdir(noisy_dir):
        if filename.endswith('.wav'):
            in_path = os.path.join(noisy_dir, filename)
            out_path = os.path.join(wiener_dir, filename)
            subprocess.run(['python', 'wiener_filter.py', '--input', in_path, '--output', out_path, '--noise-duration', '0.25'])
            
print('Finished applying the Wiener filter!')
