

from tensorflow.keras.models import load_model
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
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
    if not os.path.exists(path):
        os.mkdir(path)


def clean_and_predict(args):
    """Predict for each dt of a wav file."""

    src_root = args.src_root
    clean_dst = args.clean_dst
    dt = args.delta_time
    wav_fn = "".join(args.vid_fn.split(".")[:-1]) + ".wav"


    # Cleaning
    rate, wav = downsample_mono(os.path.join(src_root, wav_fn), args.sr)
    mask, y_mean = envelope(wav, rate, threshold=args.threshold)
    wav = wav[mask]
    delta_sample = int(dt*rate)

    # cleaned audio is less than a single sample
    # pad with zeros to delta_sample size
    if wav.shape[0] < delta_sample:
        sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
        sample[:wav.shape[0]] = wav
        # Predict on each sample

        """Sample, rate, sr"""

    # step through audio and save every delta_sample
    # discard the ending audio if it is too short
    else:
        for cnt, i in enumerate(np.arange(0, wav.shape[0], delta_sample)):
            start = int(i)
            stop = int(i + delta_sample)
            sample = wav[start:stop]


class Predictor:
    def __init__(self, args):
        """Load the model and other prediction vars."""
        # Predictions
        self.args = args
        self.threshold = 0.98
        print(f"Predicting with threshold = {self.threshold}")
        self.predictions = []
        self.model = load_model(args.model_fn,
            custom_objects={'STFT':STFT,
                            'Magnitude':Magnitude,
                            'ApplyFilterbank':ApplyFilterbank,
                            'MagnitudeToDecibel':MagnitudeToDecibel})

    def clean_and_predict(self):
        """Predict for each dt of a wav file."""
        src_root = self.args.src_root
        clean_dst = self.args.clean_dst
        dt = self.args.delta_time
        vid_fn = self.args.vid_fn
        wav_threshold = self.args.threshold
        sr = self.args.sr
        wav_fn = "".join(vid_fn.split(".")[:-1]) + ".wav"


        # Cleaning
        rate, wav = downsample_mono(os.path.join(src_root, wav_fn), sr)
        # mask, y_mean = envelope(wav, rate, threshold=wav_threshold)
        # wav = wav[mask]
        delta_sample = int(dt*rate)

        # cleaned audio is less than a single sample
        # pad with zeros to delta_sample size
        if wav.shape[0] < delta_sample:
            sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
            sample[:wav.shape[0]] = wav
            # Predict on the sample
            p = self.make_prediction(sample)
            print(p)

        # step through audio and save every delta_sample
        else:
            for cnt, i in enumerate(np.arange(0, wav.shape[0], delta_sample)):
                start = int(i)
                stop = int(i + delta_sample)
                sample = wav[start:stop]
                p = self.make_prediction(sample)
                print(p)

    def make_prediction(self, wav):
        """Make a prediction on a single dt."""
        print("HERE")
        step = int(self.args.sr * self.args.delta_time)
        batch = []

        for i in range(0, wav.shape[0], step):
            sample = wav[i:i+step]
            sample = sample.reshape(-1, 1)
            if sample.shape[0] < step:
                tmp = np.zeros(shape=(step, 1), dtype=np.float32)
                tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
                sample = tmp
            batch.append(sample)
        X_batch = np.array(batch, dtype=np.float32)
        print(len(X_batch))
        y_pred = self.model.predict(X_batch)
        y_mean = np.mean(y_pred, axis=0)
        ## y_pred[0][0] -> laugh
        ## y_pred[0][1] -> not_laugh
        print(y_mean)
        y_pred = int(y_pred[0][0] > self.threshold)


        # X_batch = np.array(batch, dtype=np.float32)
        # y_pred = self.model.predict(X_batch)
        # y_mean = np.mean(y_pred, axis=0)
        # ## y_mean[0]-> laugh
        # ## y_mean[1] -> not_laugh
        # prediction = int(y_mean[0] > self.threshold)
        # print(y_pred)
        # print(y_mean)
        # return prediction
