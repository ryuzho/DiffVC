# DiffVC
## Diffusion Model for Voice Conversion

      1. requirements.txt module install

      2. Preprocessing
         ㄴpython preprocess.py /path/to/dir/containing/wavs

      ex) python preprocess.py DiffVC/sample/wavs

       실행결과 DiffVC/sample/wavs 폴더에 {filename}.wav.spec.npy 생성
   

      3. Training
      ㄴpython __main.py__ /path/to/model/dir /path/to/dir/containing/wavs

      ex) python __main.py__ DiffVC/model_dir DiffVC/sample/wavs
  
       ** {model_dir} 폴더 먼저 생성 후 __main.py__ 진행

      4. Inference
      ㄴpython inference.py /path/to/model /path/to/source_wav /path/to/target_spectrogram

      ex) python inference.py DiffVC/sample/inference_sample DiffVC/sample/inference_sample/LJ001-0103.wav DiffVC/sample/inference_sample/LJ001-0025.wav.spec.npy 
