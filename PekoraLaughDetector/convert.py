import ffmpeg
from clean import get_first_filename
import os

def convert_vid_to_wav(src_root="video_input", vid_fn = None):
    valid_extensions = (".mp4", ".mkv")
    if not vid_fn:
        vid_fn = get_first_filename(src_root, valid_extensions)

    vid_path = os.path.join(src_root, vid_fn)

    print(vid_path)

    input_vid = ffmpeg.input(vid_path)
    output_wav = ffmpeg.output(input_vid, vid_path.split(".")[0] + ".wav")
    output_wav.run()
