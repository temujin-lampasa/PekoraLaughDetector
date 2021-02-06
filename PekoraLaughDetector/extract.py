from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from clean import get_first_filename, check_dir
import os


def extract(args):
    """Extract the video segments with positive predictions."""
    print("Extracting ...")
    valid_extensions = args.valid_extensions
    src_root = args.src_root
    extract_dst = args.extract_dst
    pred_file = args.pred_file
    check_dir(src_root)
    check_dir(extract_dst)
    src_fn = get_first_filename(src_root, valid_extensions)

    output_extension = "." + src_fn.split(".")[-1]

    # Retrieve predictions
    predictions = None
    with open(pred_file, 'r') as p:
        predictions = [int(i) for i in p.readlines()[0].strip()]

    # Get segments with positive predictions
    segments = segment_array(predictions)

    # Extract subclips
    clip_num = 0
    for start, end in segments:
        clip_num += 1
        targetname = f"video_output/laugh{clip_num}{output_extension}"
        ffmpeg_extract_subclip(os.path.join(src_root, src_fn),
        start, end, targetname=targetname)

def segment_array(array, patience=3, positive=1, min_size=0,
left_padding=0, right_padding=1):
    """Divide an array into segments of positive instances.
    Used for retrieving segments [seg_start, seg_end) from an input video.

    Args:
        patience (int): max space between segments.
        positive (int, str): the positive class.
        min_size (int): minimum segment size before padding.
        left_padding (int): padding for left side of segment.
        right_padding (int): padding for right side of segment.
    Returns:
        A list of 2-tuples (start_of_segment, end_of_segment).

    Example:
        For:
            array = [1, 0, 0, 1, 0, 0, 0, 0, 1]
        Calling this function:
            segment_array(array, patience=3, left_padding=0,
                          right_padding=0, min_size=0)
        Returns this value:
            [(0, 4), (8, 9)]
    """
    seg_start = []  # start second
    seg_end = []  # end second
    in_segment = False
    steps_since_positive = 0

    print("Retrieving segments...")
    for index, value in enumerate(array):
        if index != len(array) - 1: # if not last elem
            if in_segment:
                if value == positive:  # restart step count
                    steps_since_positive = 0
                else:
                    steps_since_positive += 1
                    if steps_since_positive > patience:  # out of patience, end segment
                        in_segment = False
                        seg_end.append(index - patience)
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
                    if steps_since_positive > patience:
                        seg_end.append(index - patience)
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

    # Add left and right buffers
    for index, seg in enumerate(min_segments):
        start = seg[0] - left_padding
        if start < 0:  # don't let start index be negative
            start = 0
        end = seg[1] + right_padding
        min_segments[index] = (start, end)

    # Merge overlapping segments
    segments_merged = []
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
                segments_merged.append(current_segment)
                current_segment = next_segment
    # When you run out of segments, flush the current_segment
    if current_segment:
        segments_merged.append(current_segment)

    return segments_merged
