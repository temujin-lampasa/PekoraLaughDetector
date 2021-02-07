import ffmpeg
import os

def convert_vid_to_wav(args):
    src_root= args.src_root
    sr = args.sr
    vid_fn = args.vid_fn
    valid_extensions = args.valid_extensions
    wav_path = os.path.join(src_root, "".join(vid_fn.split(".")[:-1]) + ".wav")
    vid_path = os.path.join(src_root, vid_fn)

    print(f"Converting {vid_path} to wav...")

    input_vid = ffmpeg.input(vid_path)
    output_wav = ffmpeg.output(input_vid, wav_path, ar=sr)
    output_wav.run()
    return wav_path
