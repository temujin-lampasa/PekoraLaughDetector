import matplotlib.pyplot as plt
from scipy.io import wavfile
import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
from librosa.core import resample, to_mono
from tqdm import tqdm
import wavio


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


def downsample_mono(path, sr):
    obj = wavio.read(path)
    wav = obj.data.astype(np.float32, order='F')
    rate = obj.rate
    try:
        channel = wav.shape[1]
        if channel == 2:
            wav = to_mono(wav.T)
        elif channel == 1:
            wav = to_mono(wav.reshape(-1))
    except IndexError:
        wav = to_mono(wav.reshape(-1))
        pass
    except Exception as exc:
        raise exc
    wav = resample(wav, rate, sr)
    wav = wav.astype(np.int16)
    return sr, wav


def save_sample(sample, rate, target_dir, fn, ix):
    fn = fn.split('.wav')[0]
    dst_path = os.path.join(target_dir.split('.')[0], fn+'_{}.wav'.format(str(ix)))
    if os.path.exists(dst_path):
        return
    wavfile.write(dst_path, rate, sample)


def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)


def split_wavs(args):
    src_root = args['src_root']
    dst_root = args['dst_root']
    check_dir(src_root)
    check_dir(dst_root)
    dt = args['delta_time']


    dirs = os.listdir(src_root)

    target_dir = dst_root

    files = [i for i in os.listdir(src_root) if i!= ".DS_Store"]
    for fn in tqdm(files):
        src_fn = os.path.join(src_root, fn)
        rate, wav = downsample_mono(src_fn, args['sr'])
        mask, y_mean = envelope(wav, rate, threshold=args['threshold'])
        # wav = wav[mask]
        delta_sample = int(dt*rate)

        # cleaned audio is less than a single sample
        # pad with zeros to delta_sample size
        if wav.shape[0] < delta_sample:
            sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
            sample[:wav.shape[0]] = wav
            save_sample(sample, rate, target_dir, fn, 0)
        # step through audio and save every delta_sample
        # discard the ending audio if it is too short
        else:
            trunc = wav.shape[0] % delta_sample
            for cnt, i in enumerate(np.arange(0, wav.shape[0]-trunc, delta_sample)):
                start = int(i)
                stop = int(i + delta_sample)
                sample = wav[start:stop]
                save_sample(sample, rate, target_dir, fn, cnt)


class Cleaner:

    def __init__(self, args):
        self.args = args

    def clean(self):
        split_wavs(self.args)
