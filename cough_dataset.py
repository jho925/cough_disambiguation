from torch.utils.data import DataLoader, Dataset, random_split
from preprocessing import spectro_gram, open_audio, resample, rechannel, pad_trunc
import librosa
import librosa.display
import matplotlib.pyplot as plt


class cough_dataloader(Dataset):
  def __init__(self, df):
    self.df = df
    self.channel = 2
    self.sr = 44100
    self.duration = 4000

  def __len__(self):
    return len(self.df)    
    
  def __getitem__(self, idx):
    audio_file =  self.df.loc[idx, 'path']
    patient_id = self.df.loc[idx, 'id']
    aud = open_audio(audio_file)
    reaud = resample(aud, self.sr)
    rechan = rechannel(reaud, self.channel)
    dur_aud = pad_trunc(rechan, self.duration)
    sgram = spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)
    return sgram, patient_id