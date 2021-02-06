from clean import clean, check_dir
from predict import make_prediction
from extract import extract
from merge import merge_clips
from convert import convert_vid_to_wav
import os

import argparse


if __name__ == '__main__':

    """
    TODO: > More training data (specifically voice)!
    """

    parser = argparse.ArgumentParser(description="Extract laugh segments from video.")

    # All
    parser.add_argument('--src_root', type=str, default='video_input/',
    help='Video source directory.', metavar='')
    parser.add_argument('--extract_dst', type=str, default='video_output/',
    help='Output video directory.', metavar='')

    # Cleaner
    parser.add_argument('--clean_dst', type=str, default='video_input/wavfile_clean',
    help='The directory where cleaned data is stored.', metavar='')
    parser.add_argument('--delta_time', type=float, default=1.0,
    help='Length of a frame. (Train a new model before changing this.)', metavar='')
    parser.add_argument('--sr', type=int, default=16_000,
    help='Sampling rate', metavar='')

    # Predict
    parser.add_argument('--model_fn', type=str, default='model/pekora_laugh_lstm.h5',
    help='Model filename', metavar='')
    parser.add_argument('--pred_fn', type=str, default='y_pred',
    help='Prediction function', metavar='')
    parser.add_argument('--threshold', type=int, default=20,
    help='Mask threshold', metavar='')
    parser.add_argument('--pred_file', type=str, default='model/predictions.txt',
    help='Prediction file path', metavar='')
    parser.add_argument('--valid_extensions', type=tuple, default=('.mp4', '.mkv'),
    help='Accepted video extensions', metavar='')

    # convert
    parser.add_argument('--vid_fn', type=str, default=None,
    help='Source video filename', metavar='')
    parser.add_argument('--no_merge', default=True, action='store_true',
    help="Don't merge the output clips")

    args, _ = parser.parse_known_args()

    check_dir(args.src_root)
    check_dir(args.extract_dst)
    check_dir(args.clean_dst)

    convert_vid_to_wav(args)
    clean(args)
    make_prediction(args)
    extract(args)
    if not args.no_merge:
        merge_clips(args)
