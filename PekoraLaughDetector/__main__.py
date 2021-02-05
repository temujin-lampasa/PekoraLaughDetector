from clean import Cleaner
from predict import Predictor
from extract import Extractor
from combine import combine_clips
from convert import convert_vid_to_wav
import os

import argparse


if __name__ == '__main__':

    """
    Instructions:
    1. Clean the input and put it in the output folder.
    2. Predict on the output folder and store the result in a list.

    TODO:
        > gather training data as well
        > Sort the clips before concatenating
        > Argparse for main
            * Add option to not extract immediately
        > For extraction:
            * Add
              * right/left buffer  -- combine overlapping
              * min_size (probably (1 or 2) + right_buffer + left_buffer)
        > Add valid_extensions arg
    """

    parser = argparse.ArgumentParser(description="Extract laugh segments from video.")

    # All
    parser.add_argument('--src_root', type=str, default='video_input/')
    parser.add_argument('--extract_dst', type=str, default='video_output/')

    # Cleaner
    parser.add_argument('--clean_dst', type=str, default='video_input/wavfile_clean')
    parser.add_argument('--delta_time', type=float, default=1.0)
    parser.add_argument('--sr', type=int, default=16_000)

    # Predict
    parser.add_argument('--model_fn', type=str, default='model/pekora_laugh_lstm.h5')
    parser.add_argument('--pred_fn', type=str, default='y_pred')
    parser.add_argument('--threshold', type=int, default=20)
    parser.add_argument('--pred_file', type=str, default='predictions.txt')

    args, _ = parser.parse_known_args()



    wav_clean_path = 'video_input/wavfile_clean'
    vid_input_path = 'video_input/'
    vid_output_path = 'video_output/'
    current_model = 'model/pekora_laugh_lstm.h5'


    cleaner = Cleaner(args)
    predictor = Predictor(args)
    extractor = Extractor(args)


    convert_vid_to_wav(vid_input_path)
    cleaner.clean()
    predictor.predict()
    extractor.extract()
    combine_clips()
