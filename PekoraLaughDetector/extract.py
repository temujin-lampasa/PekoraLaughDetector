class Extractor:
    def __init__(self):
        pass

    def extract(self):
        predictions = None
        with open("predictions.txt", 'r') as p:
            predictions = [int(i) for i in p.readlines()[0].strip()]

        segments = self.segment_array(predictions)


    def segment_array(self, array, patience=3, positive=0):
        """Divide the array into segments.
        patience = max space between segments

        for clipping interval [seg_start, seg_end)
        Ex: (1, 4)
        Starts at second 1, ends right before second 4
        """
        seg_start = []  # start second
        seg_end = []  # end second
        in_segment = False
        steps_since_positive = 0


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
        assert len(seg_start) == len(seg_end), "error segmenting the array"
        return zip(seg_start, seg_end)
