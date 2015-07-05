# A FrameSet represents a collection of frames for a named stream (e.g.
# "frame-gray", "static-flow-x") and a named process (e.g. "normal", "rotate3")
class FrameSet:
    def __init__(self, frames, streamName, processName, labels, isTestSet = False):
        self.frames = frames
        self.streamName = streamName
        self.processName = processName
        self.labels = labels
        self.isTestSet = isTestSet

    def map(self, f):
        return FrameSet(map(f, self.frames), self.streamName, self.processName, self.labels, self.isTestSet)

    def newStream(self, frames, newStreamName, newLabels=None):
        labels = newLabels if newLabels else self.labels
        return FrameSet(frames, newStreamName, self.processName, labels, self.isTestSet)

    def newProcess(self, frames, newProcessName):
        return FrameSet(frames, self.streamName, newProcessName, self.labels, self.isTestSet)

    def newFrames(self, frames):
        return FrameSet(frames, self.streamName, self.processName, self.labels, self.isTestSet)

    def markAsTest(self, isTestSet):
        self.isTestSet = isTestSet

    def getDbPostfix(self):
        return "test" if self.isTestSet else "train"

    def isFlow(self):
        return self.streamName.startswith("flow")

    def crossWith(self, otherFrameSet):
        def cross(frames1, frames2):
            crossed_length = frames1.shape[0] + frames2.shape[0]
            crossed_shape = (crossed_length,) + frames1.shape[1:]
            return np.hstack((frames1, frames2)).reshape(crossed_shape)

        crossed_frames = cross(self.frames, otherFrameSet.frames)
        crossed_labels = cross(self.labels, otherFrameSet.labels)    
        crossed_streamName = self.streamName + "-X-" + otherFrameSet.streamName
        return self.newStream(crossed_frames, crossed_streamName, crossed_labels)
