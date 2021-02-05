import os
from moviepy.editor import VideoFileClip, concatenate_videoclips


def combine_clips(args):
    src_path = args.extract_dst
    dst_path = args.extract_dst
    valid_extensions = args.valid_extensions
    output_extension = '.mp4'
    start_fname = "laugh"
    extension = None
    output_filename = "all_laughs"

    videos = []
    src_vid_files = os.listdir(src_path)

    c = 0
    while os.path.exists(os.path.join(dst_path, output_filename)):
        c += 1
        output_filename += str(c)


    for file in src_vid_files:
        extension = "." + file.split(".")[-1]
        if extension in valid_extensions and file.startswith(start_fname):
            videos.append(file)

    videos.sort(key = lambda x: int(x.split(".")[0].strip(start_fname)))

    clips = [VideoFileClip(os.path.join(src_path, v)) for v in videos]
    combined_clips = concatenate_videoclips(clips)

    combined_clips.write_videofile(os.path.join(dst_path, output_filename + output_extension))

    # Delete source clips
    for file in src_vid_files:
        os.remove(os.path.join(src_path, file))
