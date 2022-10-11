import numpy as np
import os
import torch
import torchaudio

from argparse import ArgumentParser
from params import AttrDict, params as base_params
from model import DiffVC

device = torch.device('cuda')
models = {}
fast = False

def predict(spectrogram=None,audio=None, model_dir=None, params=None, device=torch.device('cuda'), fast_sampling=False):
  
    if not model_dir in models:
        if os.path.exists(f'{model_dir}/weights.pt'):
            checkpoint = torch.load(f'{model_dir}/weights.pt')
        else:
            checkpoint = torch.load(model_dir)
        model = DiffVC(AttrDict(base_params)).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        models[model_dir] = model

    model = models[model_dir]
    model.params.override(params)
    with torch.no_grad():
        training_noise_schedule = np.array(model.params.noise_schedule)
        inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

        talpha = 1 - training_noise_schedule
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                    T.append(t + twiddle)
                    break
        T = np.array(T, dtype=np.float32)

        if not model.params.unconditional:
            if len(spectrogram.shape) == 2:# Expand rank 2 tensors by adding a batch dimension.
                spectrogram = spectrogram.unsqueeze(0)
            spectrogram = spectrogram.to(device)
            crop_mel_size = ((((audio.shape[-1]+1)//160)+1)*5)//8
            noisy_spectrogram = torch.randn(spectrogram.shape[0],model.params.n_mels,crop_mel_size, device=device)
            
        else:
            print("error")
            audio = torch.randn(1, params.audio_len, device=device)
            
        noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)
        
        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n]**0.5
            c2 = beta[n] / (1 - alpha_cum[n])**0.5
            noisy_spectrogram = c1 * (noisy_spectrogram - c2 * model(noisy_spectrogram, torch.tensor([T[n]], device=noisy_spectrogram.device), spectrogram, audio).squeeze(1))
            
            if n > 0:
                noise = torch.randn_like(noisy_spectrogram)
                sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                noisy_spectrogram += sigma * noise
                print(f" n : {n} // max : {noisy_spectrogram.max()} // mean : {noisy_spectrogram.mean()} // min : {noisy_spectrogram.min()}" )
            noisy_spectrogram = torch.clamp(noisy_spectrogram, -1.0, 1.0)
        
        return noisy_spectrogram, model.params.sample_rate



def main(args):
    if args.target_spectrogram_path:
        spectrogram = torch.from_numpy(np.load(args.target_spectrogram_path))
        audio, _ = torchaudio.load(args.source_wav_path)
        audio = audio.to(device)
        
    else:
        spectrogram = None
    mel_result, sr = predict(spectrogram, audio, model_dir=args.model_dir, fast_sampling=fast, params=base_params)
    #torchaudio.save(args.output, audio.cpu(), sample_rate=sr)
    mel = mel_result.cpu()
    
    print()
    print(" *** Synthesized mel-spectrogram *** ")
    print(mel)
    
    return mel, sr

if __name__ == '__main__':
    parser = ArgumentParser(description='inference for DiffVC')
    parser.add_argument('model_dir', help='directory containing pretrained DiffVC model')
    parser.add_argument('source_wav_path', help='path for source wav')
    parser.add_argument('target_spectrogram_path', help='path for target spectrogram')
    main(parser.parse_args())
  