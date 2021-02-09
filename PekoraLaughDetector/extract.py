import os
import shutil
from predict import check_dir
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


class Extractor:
    def __init__(self, args):
        # Make subclips directory
        self.subclips_dir = os.path.join(
            args.extract_dst,
            "subclips_" + "".join(args.vid_fn.split(".")[:-1]))
        check_dir(self.subclips_dir)

        self.subclip_fn = "subclip"

        self.src_root = args.src_root
        self.extract_dst = args.extract_dst
        self.pred_file =  args.pred_file
        self.vid_fn = args.vid_fn
        self.keep = args.keep



    def extract(self):
        """Extract the video segments with positive predictions."""
        print("Extracting ...")
        output_extension = "." + self.vid_fn.split(".")[-1]

        # Retrieve predictions
        predictions = None
        with open(self.pred_file, 'r') as p:
            predictions = [int(i) for i in p.readlines()[0].strip()]

        # Get segments with positive predictions
        segments = segment_array(predictions)

        # Extract subclips to directory
        clip_num = 0
        for start, end in segments:
            clip_num += 1
            targetname = os.path.join(
                self.subclips_dir,
                f"{self.subclip_fn}{clip_num}{output_extension}")
            src_file = os.path.join(self.src_root, self.vid_fn)
            ffmpeg_extract_subclip(src_file, start, end, targetname=targetname)

    def merge_clips(self):
        output_fn = "".join(self.vid_fn.split(".")[:-1]) + ".mp4"

        # merge subclips
        subclips_fn = os.listdir(self.subclips_dir)
        subclips_fn.sort(key=lambda x: int(x.split(".")[0].strip(self.subclip_fn)))
        subclips = [VideoFileClip(os.path.join(self.subclips_dir, sc)) for sc in subclips_fn]
        combined_clips = concatenate_videoclips(subclips)
        combined_clips.write_videofile(os.path.join(self.extract_dst, output_fn))

        # Delete subclips after merging
<<<<<<< HEAD
        if not self.keep:
=======
        if not args.keep():
>>>>>>> 51407c45ffd33b4895a4e67caf9b04c257aedc5e
            shutil.rmtree(self.subclips_dir)


def segment_array(array, tolerance=3, positive=1, min_size=0,
left_padding=0, right_padding=1):
    """Divide an array into segments of positive instances.
    Used for retrieving segments [seg_start, seg_end) from an input video.

    Args:
        tolerance (int): max space between segments.
        positive (int, str): the positive class.
        min_size (int): minimum segment size before padding.
        left_padding (int): padding for left side of segment.
        right_padding (int): padding for right side of segment.

    Returns:
        A list of 2-tuples (start_of_segment, end_of_segment).

    Example:
        For array = [1, 0, 0, 1, 0, 0, 0, 0, 1]
        Calling this function:
            segment_array(array, tolerance=3, left_padding=0,
                          right_padding=0, min_size=0)
        Returns this value:
            [(0, 4), (8, 9)]
    """
    seg_start = []
    seg_end = []
    in_segment = False
    steps_since_positive = 0

    print("Retrieving segments...")
    for index, value in enumerate(array):
        if index != len(array) - 1: # if not last elem
            if in_segment:
                if value == positive:
                    steps_since_positive = 0
                else:
                    steps_since_positive += 1
                    if steps_since_positive > tolerance:  # end segment
                        in_segment = False
                        seg_end.append(index - tolerance)
            elif not in_segment:
                if value == positive:  # start of segment
                    steps_since_positive = 0
                    in_segment = True
                    seg_start.append(index)
                else:
                    steps_since_positive += 1
        else: #if last elem.
            if in_segment:
                if value == positive:
                    steps_since_positive = 0
                    seg_end.append(index + 1)
                else:
                    steps_since_positive += 1
                    if steps_since_positive > tolerance:
                        seg_end.append(index - tolerance)
                    else:
                        seg_end.append(index - steps_since_positive + 1)
            elif not in_segment:
                if value == positive:
                    seg_start.append(index)
                    seg_end.append(index + 1)
                else:
                    continue

    # Remove all segments less than min size
    min_segments = []

    for start, end in zip(seg_start, seg_end):
        if (end - start) >= min_size:
            min_segments.append((start, end))

    # Add left and right padding
    for index, seg in enumerate(min_segments):
        start = seg[0] - left_padding
        if start < 0:
            start = 0
        end = seg[1] + right_padding
        if end > len(array):
            end = len(array)
        min_segments[index] = (start, end)

    # Merge overlapping segments
    merged_segments = []
    current_segment = None

    min_segments.sort()
    while min_segments:
        if current_segment is None:  # first iter, set curr_segment
            current_segment = min_segments.pop(0)
        else:  # compare curr_segment to next segment
            next_segment = min_segments.pop(0)
            if current_segment[1] >= next_segment[0]:  # merge
                current_segment = (current_segment[0], next_segment[1])
            else:  # pop and change the current segment
                merged_segments.append(current_segment)
                current_segment = next_segment
    if current_segment:
        merged_segments.append(current_segment)

    # Connect close together segments
    conn_segments = []
    while merged_segments:
        next = merged_segments.pop(0)
        if len(conn_segments) == 0:
            conn_segments.append(next)
        else:
            latest_seg = conn_segments[-1]
            if (next[0] - latest_seg[1]) < tolerance:
                new_seg = (latest_seg[0], next[1])
                conn_segments.pop()
                conn_segments.append(new_seg)
            else:
                conn_segments.append(next)

    return conn_segments
