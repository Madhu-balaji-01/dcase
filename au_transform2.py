import torch
from torch import tensor
from torchvision import transforms
# import torchaudio
import torchaudio as torch_audio
# import requests
import matplotlib.pyplot as plt
import time

class Audio_Transform:

    class DataNode:
        def __init__(self, data, fs, time):
            self.data = data
            self.fs = fs
            self.time = time

    def __init__(self, mono=None, spectra_type=None, device=None, para=None):
        self.mono = mono
        self.spectra_type = spectra_type
        self.device = device
        self.fs = para['fs']
        self.time= para['time']
        self.n_fft = para['n_fft']
        self.win_length = para['win_length']
        self.hop_length = para['hop_length']
        # self.ip = None
        if self.spectra_type is 'Mel_Spectrum':
            self.spectrum = self.trans_melspectrogram()
        else:
            self.spectrum = self.trans_spectrogram()
        self.am_to_db = self.trans_am_to_db()
        self.au_to_img = self.trans_autoimg()
        # self.normalise = self.trans_normalize()

        torch_audio.set_audio_backend(backend="sox_io")
        self.spectrum = self.spectrum.to(self.device)
        self.am_to_db = self.am_to_db.to(self.device)
        # self.au_to_img = self.au_to_img.to(self.device)
        # self.normalise = self.normalise.to(self.device)

    def trans_autoimg(self):
        img_transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.ToTensor()])
        return img_transform

    def trans_spectrogram(self):
        spectrum = torch_audio.transforms.Spectrogram(n_fft=self.n_fft,
                                                     win_length=self.win_length,
                                                     hop_length=self.hop_length,
                                                     normalized=True)
        return spectrum

    def trans_melspectrogram(self):
        spectrum = torch_audio.transforms.MelSpectrogram(sample_rate=self.fs,
                                                        win_length=self.win_length,
                                                        hop_length=self.hop_length,
                                                        n_fft=self.n_fft,
                                                        f_min=0,
                                                        f_max= self.fs/2,
                                                        n_mels=500,
                                                        normalized=True)

        return spectrum

    def trans_am_to_db(self):
        return torch_audio.transforms.AmplitudeToDB()

    def trans_normalize(self, ip):
        # Subtract the mean, and scale to the interval [-1,1]
        ip_min_mean = ip - ip.mean(dim=1)[:, None]
        ip_norm = ip_min_mean / ip_min_mean.max(dim=1).values[:, None]
        return ip_norm

    def read_raw_wav(self, filename_list, label_list):
        raw_au_dict = {}
        labels = []
        for i, file in enumerate(filename_list):
            try:
                ip, fs = torch_audio.load(file)
                raw_au_dict[str(i)] = self.DataNode(ip, fs, ip.shape[1] / fs)
                labels.append(label_list[i])
            except:
                print('audio not loaded', file)
                pass
        return raw_au_dict, labels

    def rawau_to_tensor(self, raw_au_dict):

        ip = torch.empty(1, self.time * self.fs)
        resamp = torch_audio.transforms.Resample(orig_freq=44100, new_freq=self.fs)

        for key in raw_au_dict:
            aud = raw_au_dict[key]
            # resampling of audio data
            if aud.fs == 44100:
                aud.data = (aud.data)
            elif aud.fs == 16000:
                pass
            else:
                aud.data = torch_audio.transforms.Resample(orig_freq=aud.fs, new_freq=self.fs)(aud.data)
            # fixing audio data size
            if aud.time > self.time:
                aud.data = aud.data[:, 0:self.fs * self.time]
            elif aud.time < self.time:
                if aud.time >= self.time / 2:
                    req_extra_data = (self.fs * self.time) - aud.data.shape[1]
                    aud.data = torch.hstack((aud.data, aud.data[:, 0:req_extra_data]))
                else:
                    while aud.time < self.time / 2:
                        aud.data = torch.hstack((aud.data, aud.data))
                        aud.time = aud.time * 2
                    req_extra_data = (self.fs * self.time) - aud.data.shape[1]
                    aud.data = torch.hstack((aud.data, aud.data[:, 0:req_extra_data]))
            # data matrix
            if key == '0':
                ip = aud.data
            else:
                ip = torch.vstack((ip, aud.data))

        return ip

    def label_to_torch(self,label_list):
        label = torch.Tensor(label_list)
        label = label.to(self.device)
        return label

    def normalize_spectra(self, x):
        # (b, w, h) = x.shape
        # x_norm = x
        # x_norm = x_norm.view(x_norm.size(0), -1)
        # x_norm -= x_norm.min(1, keepdim=True)[0]
        # x_norm /= x_norm.max(1, keepdim=True)[0]
        # x_norm = x_norm.view(b, w, h)

        min_val1 = torch.min(x, dim=1, keepdim=True)[0]
        min_val2 = torch.min(min_val1, dim=2, keepdim=True)[0]

        x_step1 = torch.sub(x, min_val2)

        max_val1 = torch.max(x_step1, dim=1, keepdim=True)[0]
        max_val2 = torch.max(max_val1, dim=2, keepdim=True)[0]

        x_norm = x_step1 / max_val2

        return x_norm

    def audio_to_img(self, spectra):
        img = None
        for i in range(spectra.shape[0]):
            if i == 0:
                temp = self.au_to_img(spectra[i, :, :])
                img = temp
            else:
                temp = self.au_to_img(spectra[i, :, :])
                img = torch.vstack((img, temp))
        # img_trans = img[:, None, :, :]
        return img

    def main(self, filename_list, label_list):
        # process in cpu
        # start_time = time.time()
        raw_au_dict, labels = self.read_raw_wav(filename_list, label_list)
        ip = self.rawau_to_tensor(raw_au_dict)
        ip_norm = self.trans_normalize(ip)

        # send the data to gpu
        ip_norm = ip_norm.to(self.device, dtype=torch.float32)

        # process in gpu
        spectra = self.spectrum(ip_norm)
        spectra_db = self.am_to_db(spectra)
        spectra_db_norm = self.normalize_spectra(spectra_db)
        spectra_db_norm2 = torch.stack((spectra_db_norm[0::2].detach().clone(),spectra_db_norm[1::2].detach().clone()), dim=0)
        if self.mono == 'mean':
            spectra_db_norm = torch.mean(spectra_db_norm2, dim=0)
        elif self.mono == 'diff':
            spectra_db_norm = (spectra_db_norm2[0] - spectra_db_norm2[1])/2
        spectra_img = self.audio_to_img(spectra_db_norm)

        # send the data to gpu
        spectra_img = spectra_img.to(self.device, dtype=torch.float32)
        spectra_img = spectra_img[:,None,:,:]
        # spectra_img = torch.rot90(spectra_img, 3, [2, 3])

        label = self.label_to_torch(labels)
        # print('audio transform time', time.time()-start_time)
        
        return spectra_img, label


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    filename_list = ['/home/audio_server1/intern/dataset/dcase/audio/airport-helsinki-3-112.wav']
    label_list = [0]
    para = {}
    para['fs'] = 16000
    para['time'] = 10
    para['n_fft'] = 1024
    para['win_length'] = int(0.05 * 16000)
    para['hop_length'] = int(0.02 * 16000)

    transform = Audio_Transform(spectra_type='Spectrum', device=device, para=para)

    x, y = transform.main(filename_list, label_list)