import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
from clean import check_dir


def merge_clips(args):
    output_extension = '.mp4'
    start_fname = "laugh"
    dst_root = args.extract_dst


    # Make subclips directory
    subclips_dir = os.path.join(args.extract_dst, "subclips_" + args.vid_fn)
    check_dir(subclips_dir)

    # merge subclips
    subclips_fn = os.listdir(subclips_dir)
    subclips_fn.sort(key = lambda x: int(x.split(".")[0].strip(start_fname)))
    subclips = [VideoFileClip(os.path.join(subclips_dir, sc)) for sc in subclips_fn]
    combined_clips = concatenate_videoclips(subclips)
    combined_clips.write_videofile(os.path.join(dst_root, args.vid_fn))

    # Delete subclips after merging
    for file in subclips_dir:
        os.remove(os.path.join(src_path, file))
