import os
import numpy as np
from tqdm import tqdm
import wavio
from scipy.io import wavfile
from librosa.core import resample, to_mono
from tensorflow.keras.models import load_model
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel


class Predictor:
    def __init__(self, args):
        """Load the model and other prediction vars."""
        self.args = args
        self.pred_threshold = 0.99
        self.predictions = []
        self.model = load_model(args.model_fn,
            custom_objects={'STFT':STFT,
                            'Magnitude':Magnitude,
                            'ApplyFilterbank':ApplyFilterbank,
                            'MagnitudeToDecibel':MagnitudeToDecibel})

    def clean_and_predict(self):
        """Predict for each dt of a wav file. Store in self.predictions."""
        print(f"Predicting with threshold = {self.pred_threshold} ...")
        args = self.args
        wav_fn = "".join(args.vid_fn.split(".")[:-1]) + ".wav"
        rate, wav = downsample_mono(os.path.join(args.src_root, wav_fn), args.sr)
        delta_sample = int(args.delta_time*rate)

        # audio is less than a single sample
        # pad with zeros to delta_sample size, then predict
        if wav.shape[0] < delta_sample:
            sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
            sample[:wav.shape[0]] = wav
            p = self.make_prediction(sample)
            self.predictions.append(int(p))
        # step through audio and predict for every delta_sample
        else:
            steps = np.arange(0, wav.shape[0], delta_sample)
            for i in tqdm(steps, total=len(steps)):
                start = int(i)
                stop = int(i + delta_sample)
                sample = wav[start:stop]
                p = self.make_prediction(sample)
                self.predictions.append(int(p))

    def make_prediction(self, wav):
        """Make a prediction on a single dt."""
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
        y_pred = self.model.predict(X_batch)
        y_mean = np.mean(y_pred, axis=0)
        # y_mean[0] -- laugh
        # y_mean[1] -- not_laugh
        prediction = y_mean[0] > self.pred_threshold
        return prediction

    def save_predictions(self):
        """Save predictions to text file."""
        pred_string = "".join([str(p) for p in self.predictions])
        with open(self.args.pred_file, 'w+') as f:
            f.write(pred_string)


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


def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
