import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from clean import downsample_mono, envelope


class Predictor:
    def __init__(self, args):
        """Load the model and other prediction vars."""
        self.args = args
        self.pred_threshold = 0.98
        self.predictions = []
        self.model = load_model(args.model_fn,
            custom_objects={'STFT':STFT,
                            'Magnitude':Magnitude,
                            'ApplyFilterbank':ApplyFilterbank,
                            'MagnitudeToDecibel':MagnitudeToDecibel})

    def clean_and_predict(self):
        """Predict for each dt of a wav file. Store in self.predictions."""
        print(f"Predicting with threshold = {self.pred_threshold} ...")
        wav_fn = "".join(self.args.vid_fn.split(".")[:-1]) + ".wav"
        rate, wav = downsample_mono(
            os.path.join(self.args.src_root, wav_fn), self.args.sr
        )
        # mask, y_mean = envelope(wav, rate, threshold=self.args.threshold)
        # wav = wav[mask]
        delta_sample = int(self.args.delta_time*rate)

        # cleaned audio is less than a single sample
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
