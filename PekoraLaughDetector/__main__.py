from clean import Cleaner
from predict import Predictor
from extract import Extractor
from combine import combine_clips
from convert import convert_vid_to_wav
import os

import argparse


if __name__ == '__main__':

    """
    TODO:
        > Argparse for main
            * Add option to not extract immediately
        > For extraction:
            * Add
              * min_size (probably (1 or 2) + right_buffer + left_buffer)
                > Filter by min_size before adding the buffer !!
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
    parser.add_argument('--valid_extensions', type=tuple, default=('.mp4', '.mkv'))

    # convert
    parser.add_argument('--vid_fn', type=str, default=None)

    args, _ = parser.parse_known_args()



    cleaner = Cleaner(args)
    predictor = Predictor(args)
    extractor = Extractor(args)


    convert_vid_to_wav(args)
    cleaner.clean()
    predictor.predict()
    extractor.extract()
    combine_clips(args)
