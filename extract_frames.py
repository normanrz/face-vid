###############################################################################
#
# Use this script the extract frames from the MMI Facial Expresssion DB.
# Every n-th frame will be extracted. Frames will be processed in the following
# manner:
# - converted to grey-scale
#   - cropping to detected faces
#   - black oval mask around face
#   - save optical flow along x & y axis
#
# Usage: extract_frames.py <max_frame_count> <path_to_video> <output_path>
#
###############################################################################
from __future__ import generators
import cv2, os, sys, itertools, functools, h5py
import numpy as np
from itertools import izip

CLASSIFIER_PATH = os.path.join(os.path.dirname(sys.argv[0]), "haarcascade_frontalface_alt.xml")
SCALE_FLOW = 10
faceCascade = cv2.CascadeClassifier(CLASSIFIER_PATH)

# A FrameSet represents a collection of frames for a named stream (e.g.
# "frame-gray", "static-flow-x") and a named process (e.g. "normal", "rotate3")
class FrameSet:
    def __init__(self, frames, streamName, processName = "normal"):
        self.frames = frames
        self.streamName = streamName
        self.processName = processName

    def map(self, f):
        return FrameSet(map(f, self.frames), self.streamName, self.processName)

    def newStream(self, frames, newStreamName):
        return FrameSet(frames, newStreamName, self.processName)

    def newProcess(self, frames, newProcessName):
        return FrameSet(frames, self.streamName, newProcessName)



# Do face detection and return the first face
def detect_face(image):
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    if len(faces) == 0:
        sys.exit()

    return faces


# Special processing relevant for the MMI facial dataset
def preprocessMMI(image):
    # turn into greyscale
    imageAsGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(imageAsGray)

    imageAsYCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)  # change the color image from BGR to YCrCb format
    channels = cv2.split(imageAsYCrCb)  # split the image into channels
    channels[0] = cv2.equalizeHist(channels[0])  # equalize histogram on the 1st channel (Y)
    imageWithEqualizedHist = cv2.merge(channels)  # merge 3 channels including the modified 1st channel into one image
    imageAsBGR = cv2.cvtColor(imageWithEqualizedHist,
                              cv2.COLOR_YCR_CB2BGR)  # change the color image from YCrCb to BGR format (to display image properly)

    return (imageAsGray, imageAsBGR)


def read_video(video):
    # read video
    framesGray = []
    framesBGR = []
    cap = cv2.VideoCapture(video)
    frame_count = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    if cap.isOpened():

        for i in range(0, frame_count):
            # actually read a frame
            returnValue, frame = cap.read()

            if not returnValue:
                break

            (imageAsGray, imageAsBGR) = preprocessMMI(frame)
            framesGray.append(imageAsGray)
            framesBGR.append(imageAsBGR)

        cap.release()
        return (FrameSet(framesGray, "frame-gray"), FrameSet(framesBGR, "frame-bgr"))
    else:
        sys.exit("Error opening video file.")


# Invoke face detection, find largest cropping window and apply elliptical mask
def face_pass(framesGray, framesBGR):
    def crop_and_mask(frame, minX, minY, maxWidth, maxHeight, count):
        cropped_frame = frame[minY: minY + maxHeight, minX: minX + maxWidth]

        center = (int(maxWidth * 0.5), int(maxHeight * 0.5))
        axes = (int(maxWidth * 0.4), int(maxHeight * 0.5))

        mask = np.zeros_like(cropped_frame)
        cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)

        return np.bitwise_and(cropped_frame, mask)

    def point_in_rect(point, rect):
        return \
            point["x"] > rect["minX"] and point["x"] < rect["minX"] + rect["maxWidth"] and \
            point["y"] > rect["minY"] and point["y"] < rect["minY"] + rect["maxHeight"]

    # Remember all faces, group faces that are within the same enclosing rectangle
    def remember_face(face, known_faces):
        (x, y, w, h) = face

        if len(known_faces) == 0:
            return [{
                    "minX" : x,
                    "minY" : y,
                    "maxWidth" : w,
                    "maxHeight" : h,
                    "count" : 0
                }]
        else:
            center_point = {
                "x" : x + w / 2 ,
                "y" : y + h / 2
            }
            head, tail = known_faces[0], known_faces[1:]
            if point_in_rect(center_point, head):
                return [{
                        "minX" : min(head["minX"], x),
                        "minY" : min(head["minY"], y),
                        "maxWidth" : max(head["maxWidth"], w),
                        "maxHeight" : max(head["maxHeight"], h),
                        "count" : head["count"] + 1
                    }] + tail
            else:
                return [head] + remember_face(face, tail)


    known_faces = []
    for i, frame in enumerate(framesGray.frames):

        # only do face detection every 10 frames to save processing power
        if i % 10 <> 0:
            continue

        # Find all faces
        for face in detect_face(frame):
            known_faces = remember_face(face, known_faces)


    most_significant_face = max(known_faces, key=lambda x: x["count"])
    return (
        framesGray.map(lambda f: crop_and_mask(f, **most_significant_face)),
        framesBGR.map(lambda f: crop_and_mask(f, **most_significant_face))
    )


def calculateFlow(frame1, frame2):
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, 0.5, 3, 15, 3, 2, 1.1, 0)
    horz = cv2.convertScaleAbs(flow[..., 0], None, 128 / SCALE_FLOW, 128)
    vert = cv2.convertScaleAbs(flow[..., 1], None, 128 / SCALE_FLOW, 128)
    return horz, vert


# Increase the dataset by adding some rotated and re-colored images
def multiply_frames(frameSet):
    def rotate_frame(frame, angle):
        rows = frame.shape[0]
        cols = frame.shape[1]
        rotMat = cv2.getRotationMatrix2D((rows / 2, cols / 2), angle, 1)
        return cv2.warpAffine(frame, rotMat, (cols, rows))

    def contrast_brightness(frame, gain, bias):
        return np.array(np.fmax(np.fmin(gain * np.array(frame, dtype=np.float32) + bias, 255), 0), dtype=np.uint8)

    yield frameSet
    for i, angle in enumerate([0, -3, 3, -6, 6]):
        rotatedFrames = [rotate_frame(frame, angle) for frame in frameSet.frames]
        for j, gain in enumerate([0]):
            for k, bias in enumerate([-40, -20, 0, 20]):
                newFrames = [contrast_brightness(frame, 1.1 ** gain, bias) for frame in rotatedFrames]
                yield frameSet.newProcess(newFrames, "multi%i-%i-%i" % (i, j, k))


# Calculate the optical flow along the x and y axis
# always compares with the previous/next #x images in the series
def flow_pass(frameSet):

    sliding_window = 5

    # Find the 10 frames around each flow image and stack them as a single 10channel blob
    def stack_frames(flow_data, frame_number):

        relevant_frames = range(frame_number - sliding_window, frame_number + sliding_window)
        frames = map(lambda i: empty_frame if i < 0 or i >= len(flow_data) else flow_data[i], relevant_frames)
        return cv2.merge(frames)

    flows = zip(*[calculateFlow(f1, f2) for f1, f2 in zip(frameSet.frames[0] + frameSet.frames, frameSet.frames)])
    empty_frame = np.ones_like(flows[0][0]) * 128

    stacked_flows  = map(lambda flow:
        map(functools.partial(stack_frames, flow), range(0, len(flow)))
    , flows)

    return (
        frameSet.newStream(stacked_flows[0], "flow-x"),
        frameSet.newStream(stacked_flows[1], "flow-y")
    )


def save_to_disk(output_path, frameSets):

    for frameSet in frameSets:
        for i, frame in enumerate(frameSet.frames):
            cv2.imwrite(os.path.join(output_path, "%s_%s_%s.png" % (frameSet.processName, frameSet.streamName, i)), frame)


def save_as_hdf5():
    h5file = h5py.File("test.hdf5", "w")

    try:
	# get the datasets
	data_dataset = h5file["data"]
	label_dataset = h5file["label"]
	# set the start indices
	start_data = data_dataset.shape[-1]
	start_label = label_dataset.shape[-1]
	# resize the datasets so that the new data can fit in
	data_dataset.resize(start_data + data.shape[-1], 3)
	label_dataset.resize(start_data + labels.shape[-1], 1)
    except KeyError:
	# create new datasets in hdf5 file
	data_shape = data.shape
	data_dataset = h5file.create_dataset(
	    "/data",
	    shape=data_shape,
	    maxshape=(
		data_shape[0],
		data_shape[1],
		data_shape[2],
		None,
	    ),
	    dtype="f",
	    chunks=True,
	)
	label_shape = labels.shape
	label_dataset = h5file.create_dataset(
	    "/label",
	    shape=label_shape,
	    maxshape=(
		label_shape[0],
		None,
	    ),
	    dtype="f",
	    chunks=True,
	)
	# set the start indices in fourth dimension
	start_data = 0
	start_label = 0

    if label_dataset is not None and data_dataset is not None:
	# write the given data into the hdf5 file
	data_dataset[:, :, :, start_data:start_data + data.shape[-1]] = data
	label_dataset[:, start_label:start_label + labels.shape[-1]] = labels

    finally
	h5file.flush()
	h5file.close()


def reduce_dataset(max_frame_count, frameSets):

    # only grab & compute every x-th frame or all if count == 0
    def filter_relevant_frames(frameSet):
        frame_count = len(frameSet.frames)
        if max_frame_count > 0:
            stride = frame_count / float(max_frame_count)
        else:
            stride = 1

        filtered_frames = [frameSet.frames[int(i)] for i in np.arange(0, frame_count, stride)]
	return frameSet.newStream(filtered_frames, frameSet.streamName)

    return tuple(map(filter_relevant_frames, frameSets))


def main():
    if len(sys.argv) < 4:
        sys.exit("Usage: %s <max_frame_count> <path_to_video> <output_path>" % sys.argv[0])

    # read path to image as command argument
    max_frame_count = int(sys.argv[1])
    video_path = os.path.abspath(sys.argv[2])
    output_path = os.path.abspath(sys.argv[3])

    if not os.path.isfile(video_path):
        sys.exit("The specified <path_to_video> argument is not a valid filename")

    if not os.path.isdir(output_path):
        sys.exit("The specified <output_path> argument is not a valid directory")

    # ready to rumble
    framesGray, framesBGR = read_video(video_path)

    for framesGray, framesBGR in izip(multiply_frames(framesGray), multiply_frames(framesBGR)):

        # 1. find faces 2. calc flow 3. save to disk
        frames = face_pass(framesGray, framesBGR)
        flows = flow_pass(frames[0])

        frames = reduce_dataset(max_frame_count, frames)
        flows = reduce_dataset(max_frame_count, flows)

        save_to_disk(output_path, frames)
        #save_to_disk(output_path, flows)

    # exit
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
