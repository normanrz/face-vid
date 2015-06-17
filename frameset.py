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