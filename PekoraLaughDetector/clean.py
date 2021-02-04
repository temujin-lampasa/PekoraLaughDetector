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
        return False


def split_wavs(args):
    """Split a single file into wavs."""

    src_root = args['src_root']
    if not checkdir(src_root):
        raise Exception("Add a video file to the output folder.")

    # The src file is the first wav file found.
    # Assumes only 1 wav file in src_dir
    src_file = get_first_file(".wav")

    dst_root = args['dst_root']
    check_dir(dst_root)

    dt = args['delta_time']

    target_dir = dst_root

    rate, wav = downsample_mono(src_file, args['sr'])
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


def conv_to_wav(args):
    src_root = args['src_root']
    src_file = get_first_file((".mp4", ".wav"))



def get_first_file(root, extension):
    for file in os.listdir(root):
        if file.endswith(extension):
            return os.path.join(root, file)


class Cleaner:

    def __init__(self, args):
        self.args = args

    def clean(self):
        split_wavs(self.args)
