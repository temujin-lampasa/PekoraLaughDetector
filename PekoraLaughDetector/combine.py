import os
from moviepy.editor import VideoFileClip, concatenate_videoclips



def combine_clips():
    src_path = "video_output/"
    dst_path = "video_output/"
    output_extension = 'mp4'
    valid_extensions = ('mkv', 'mp4')
    start_fname = "vid"
    extension = None
    output_filename = "full_vid"

    videos = []
    src_vid_files = os.listdir(src_path)

    c = 0
    while os.path.exists(os.path.join(dst_path, output_filename)):
        c += 1
        output_filename += str(c)


    for file in src_vid_files:
        extension = file.split(".")[-1]
        if extension in valid_extensions and file.startswith(start_fname):
            videos.append(file)

    clips = [VideoFileClip(os.path.join(src_path, v)) for v in videos]
    combined_clips = concatenate_videoclips(clips)

    combined_clips.write_videofile(os.path.join(dst_path, "full_vid." + output_extension))

    # Delete src videos
    for file in src_vid_files:
        os.remove(os.path.join(src_path, file))
