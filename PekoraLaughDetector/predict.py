from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm


def make_prediction(wav, rate, sr, model_fn, delta_time, pred_file):
    """
    0 = not laugh
    1 = laugh
    """
    threshold = 0.98
    print(f"Predicting with threshold = {threshold}")

    model = load_model(args.model_fn,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel})
    predictions = []

        step = int(args.sr*args.delta_time)
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
        y_pred = model.predict(X_batch)
        y_mean = np.mean(y_pred, axis=0)
        ## y_pred[0][0] -> laugh
        ## y_pred[0][1] -> not_laugh
        y_pred = int(y_pred[0][0] > threshold)
        predictions.append((t_second, y_pred))

    with open(args.pred_file, "w+") as pred_file:
        pred_file.write("".join([str(p) for p in predictions]) + "\n")
