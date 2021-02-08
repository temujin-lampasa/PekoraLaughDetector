# PekoraLaughDetector
A tool to extract and compile Pekora laughs from videos.

## How to Use

1. Create a virtualenv and install all the requirements.
2. Place your videos in the `video_input/` folder.
3. Run `python PekoraLaughDetector`

The outputs are saved in the `video_output/` folder. <br>
If you don't want to more the outputs, pass the argument `--no_merge`.

## Acknowledgements
The deep learning model and other parts of the code were modified from seth814's repository: 
<a href="https://github.com/seth814/Audio-Classification">Audio Classification</a>.
