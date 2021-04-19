import numpy as np
import librosa
from scipy.io import wavfile
import torch
from torchvision import transforms

def read_audio(address):
    fs, ip = wavfile.read(address)
    ip = ip / 32768
    ip = np.array(ip, dtype=np.float32)
    return fs, ip


def wav_to_spectrogram(ip, para):
    s = librosa.stft(ip,
                     n_fft=para['n_fft'],
                     hop_length=para['hop_length'],
                     win_length=para['win_length'])
    s_log = np.abs(s)
    s_log = librosa.power_to_db(s_log, ref=np.max)

    return s_log

def normalize_spectra(x):

    min_val1 = np.min(x, axis=0)
    min_val2 = np.min(min_val1)

    x_step1 = x - min_val2

    max_val1 = np.max(x_step1, axis=0)
    max_val2 = np.max(max_val1)

    x_norm = x_step1/max_val2

    return x_norm

img_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.ToTensor()])


para = {}
para['fs'] = 16000
para['time'] = 10
para['n_fft'] = 1024
para['win_length'] = int(0.05 * 16000)
para['hop_length'] = int(0.02 * 16000)

fs, ip = read_audio('4_ten.wav')
spectra = wav_to_spectrogram(ip,para)
spectra_norm = normalize_spectra(spectra)
spectra_th = torch.tensor(spectra_norm)
spectra_torch = img_transform(spectra_th)
spectra_torch = spectra_torch[None,:,:,:]


