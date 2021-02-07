from clean import check_dir
from predict import Predictor
from extract import extract
from merge import merge_clips
from convert import convert_vid_to_wav
import os
import argparse
import shutil


if __name__ == '__main__':

    """
    TODO:
    > Combine clean and predict
        * Note: cleaning loses data.
    > Fix tuple arg. Change to single value.
    > support multiple files
    """

    parser = argparse.ArgumentParser(description="Extract laugh segments from video.")

    # All
    parser.add_argument('--src_root', type=str, default='video_input/',
    help='Video source directory.', metavar='')
    parser.add_argument('--extract_dst', type=str, default='video_output/',
    help='Output video directory.', metavar='')
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
    # convert
    parser.add_argument('--vid_fn', type=str, default=None,
    help='Source video filename', metavar='')
    parser.add_argument('--no_merge', default=False, action='store_true',
    help="Don't merge the output clips")

    args, _ = parser.parse_known_args()

    p = Predictor(args)

    valid_extensions = ('mp4', 'mkv')
    video_files = [f for f in os.listdir(args.src_root)
                   if f.split(".")[-1] in valid_extensions]
    for video in video_files:
        check_dir(args.src_root)
        check_dir(args.extract_dst)
        args.vid_fn = video
        wavfile_path = convert_vid_to_wav(args)
        p.clean_and_predict()
        p.save_predictions()
        os.remove(wavfile_path)
        extract(args)
        if not args.no_merge:
            merge_clips(args)
