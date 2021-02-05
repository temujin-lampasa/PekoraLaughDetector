from clean import Cleaner
from predict import Predictor
from extract import Extractor
from combine import combine_clips
from convert import convert_vid_to_wav
import os


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
        > Change threshold to 70 or 80 %
    """


    wav_clean_path = 'video_input/wavfile_clean'
    vid_input_path = 'video_input/'
    current_model = 'model/pekora_laugh_lstm.h5'


    cleaner_args = {
    'src_root': vid_input_path,
    'dst_root': wav_clean_path,
    'delta_time': 1.0,
    'sr': 16_000,
    'fn': '3a3d0279',  # todo: remove this
    'threshold': 20,
    }
    cleaner = Cleaner(cleaner_args)

    predictor_args = {
    'model_fn': current_model,
    'pred_fn': 'y_pred',
    'src_dir': wav_clean_path,
    'dt': 1.0,
    'sr': 16_000,
    'threshold': 20,
    }
    predictor = Predictor(predictor_args)

    extractor_args = {
    'src_root': vid_input_path,
    'pred_file': 'predictions.txt',
    }
    extractor = Extractor(extractor_args)


    convert_vid_to_wav(vid_input_path)
    # cleaner.clean()
    # predictor.predict()
    # extractor.extract()
    # combine_clips()
