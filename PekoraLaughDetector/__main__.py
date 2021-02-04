from clean import Cleaner
from predict import Predictor
from extract import Extractor


if __name__ == '__main__':

    """
    Instructions:
    1. Clean the input and put it in the output folder.
    2. Predict on the output folder and store the result in a list.
    """

    cleaner_args = {
    'src_root': 'input',
    'dst_root': 'output',
    'delta_time': 1.0,
    'sr': 16_000,
    'fn': '3a3d0279',  # todo: remove this
    'threshold': 20,
    }
    cleaner = Cleaner(cleaner_args)
    # cleaner.clean()

    predictor_args = {
    'model_fn': 'model/lstm.h5',
    'pred_fn': 'y_pred',
    'src_dir': 'output/',
    'dt': 1.0,
    'sr': 16_000,
    'threshold': 20,
    }
    predictor = Predictor(predictor_args)
    # predictor.predict()

    extractor = Extractor()
    extractor.extract()
