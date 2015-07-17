# A FrameSet represents a collection of frames for a named stream (e.g.
# "frame-gray", "static-flow-x") and a named process (e.g. "normal", "rotate3")

import numpy as np

class FrameSet:
    possible_formats = ["opencv", "caffe"]

    def __init__(self, frames, streamName, processName, labels, format = "opencv", isTestSet = False):
        self.frames = frames
        self.streamName = streamName
        self.processName = processName
        self.labels = labels
        if format in  FrameSet.possible_formats:
            self.format = format
        else:
            raise TypeError("format must be one of %s" % (",".join(FrameSet.possible_formats)))
        self.isTestSet = isTestSet

    def map(self, f):
        return FrameSet(map(f, self.frames), self.streamName, self.processName, self.labels, self.format, self.isTestSet)

    def newStream(self, frames, newStreamName, newLabels=None):
        labels = newLabels if newLabels != None else self.labels
        return FrameSet(frames, newStreamName, self.processName, labels, self.format, self.isTestSet)

    def newProcess(self, frames, newProcessName):
        return FrameSet(frames, self.streamName, newProcessName, self.labels, self.format, self.isTestSet)

    def newFrames(self, frames):
        return FrameSet(frames, self.streamName, self.processName, self.labels, self.format, self.isTestSet)

    def markAsTest(self, isTestSet):
        self.isTestSet = isTestSet

    def getDbPostfix(self):
        return "test" if self.isTestSet else "train"

    def isFlow(self):
        return self.streamName.startswith("flow")

    def withFormat(self, format):
        if format in FrameSet.possible_formats:
            return FrameSet(self.frames, self.streamName, self.processName, self.labels, format, self.isTestSet)
        else:
            raise TypeError("format must be one of %s" % (",".join(possible_formats)))

    def isInCaffeFormat(self):
        return self.format == "caffe"

    def isInOpenCVFormat(self):
        return self.format == "opencv"

    def crossWith(self, otherFrameSet):
        def cross(frames1, frames2):
            crossed_length = frames1.shape[1] + frames2.shape[1]
            crossed_shape = (frames1.shape[0], crossed_length,) + frames1.shape[2:]

            result = np.zeros(crossed_shape)
            for i in range(0, frames1.shape[0]):
                for j in range(0, frames1.shape[1]):
                    result[i, 2 * j] = frames1[i, j]
                    result[i, 2 * j + 1] = frames2[i, j]
            return result

        crossed_frames = cross(self.frames, otherFrameSet.frames)
        crossed_labels = cross(self.labels, otherFrameSet.labels)    
        crossed_streamName = self.streamName + "-X-" + otherFrameSet.streamName
        return FrameSet(crossed_frames, crossed_streamName, self.processName, crossed_labels, self.format, self.isTestSet)
