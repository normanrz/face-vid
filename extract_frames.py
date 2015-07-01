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
import cv2, os, sys, itertools, functools, h5py, random, numpy as np, xml.etree.ElementTree as ET
from natsort import natsorted
import pdb

CLASSIFIER_PATH = os.path.join(os.path.dirname(sys.argv[0]), "haarcascade_frontalface_alt.xml")
SCALE_FLOW = 10
NUMBER_OF_LABELS = 37

label_mapping = dict()
label_mapping_index = 0
faceCascade = cv2.CascadeClassifier(CLASSIFIER_PATH)

# A FrameSet represents a collection of frames for a named stream (e.g.
# "frame-gray", "static-flow-x") and a named process (e.g. "normal", "rotate3")
class FrameSet:
    def __init__(self, frames, streamName, processName, labels):
        self.frames = frames
        self.streamName = streamName
        self.processName = processName
        self.labels = labels

    def map(self, f):
        return FrameSet(map(f, self.frames), self.streamName, self.processName, self.labels)

    def newStream(self, frames, newStreamName, newLabels=None):
        labels = newLabels if newLabels else self.labels
        return FrameSet(frames, newStreamName, self.processName, labels)

    def newProcess(self, frames, newProcessName):
        return FrameSet(frames, self.streamName, newProcessName, self.labels)


def read_labels(path, length):

    def map_label(label):
        global label_mapping_index, label_mapping

        mapped_label = label_mapping.get(label)
        if mapped_label == None:
            label_mapping[label] = label_mapping_index
            mapped_label = label_mapping_index
            label_mapping_index += 1

        return mapped_label

    # Read and parse
    tree = ET.parse(path)
    root = tree.getroot()

    action_units = root.findall(".//ActionUnit")

    labels = np.zeros([length, NUMBER_OF_LABELS])
    for au in action_units:

        facs_code = au.get("Number")

        for marker in au.findall("Marker"):
            frame_number = int(marker.get("Frame")) - 1

            label = map_label(facs_code)
            labels[frame_number, label] = 1

    return labels

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
            framesGray.append(np.expand_dims(imageAsGray, axis = 2))
            framesBGR.append(imageAsBGR)

        cap.release()

        label_file = video.replace(".avi", "-oao_aucs.xml")
        labels = read_labels(label_file, len(framesBGR))

        return (FrameSet(framesGray, "frame-gray", "normal", labels), FrameSet(framesBGR, "frame-bgr", "normal", labels))
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
                if len(newFrames[0].shape) == 2:
                    yield frameSet.newProcess(np.expand_dims(newFrames, 3), "multi%i-%i-%i" % (i, j, k))
                else:
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


def save_as_hdf5(output_path, frameSet, db_name):

    try:

        done = False
        filename_counter = 0

        while not done:
            file_name = db_name + "_%s.h5" % filename_counter
            db_path = os.path.join(output_path, file_name)

            if os.path.isfile(db_path):
                # file must not be bigger than 1GB
                if os.stat(db_path).st_size > 1000000 * 1024:
                    filename_counter += 1
                else:
                    done = True
            else:
                done = True


        h5file = h5py.File(db_path)
        frames = np.concatenate(frameSet.frames, 0)
        labels = np.concatenate(frameSet.labels, 0)

        print frames.shape

        try:
            # get the datasets
            frames_dataset = h5file["data"]
            label_dataset = h5file["label"]

            # set the start indices
            start_data = frames_dataset.shape[0]
            start_label = label_dataset.shape[0]

            # resize the datasets so that the new data can fit in
            frames_dataset.resize(start_data + frames.shape[0], 0)
            label_dataset.resize(start_data + labels.shape[0], 0)

        except KeyError:
            # create new datasets in hdf5 file
            data_shape = (
                    frames.shape[0],
                    frames.shape[1],
                    frames.shape[2],
                    frames.shape[3],
                )
            frames_dataset = h5file.create_dataset(
                "data",
                shape=data_shape,
                maxshape=(
                    None,
                    data_shape[1],
                    data_shape[2],
                    data_shape[3]
                ),
                dtype="f",
                chunks=True,
                #compression="gzip"
            )

            label_shape = (
                    labels.shape[0],
                    labels.shape[1],
                )
            label_dataset = h5file.create_dataset(
                "/label",
                shape=label_shape,
                maxshape=(
                    None,
                    label_shape[1]
                ),
                dtype="f",
                chunks=True,
                #compression="gzip"
            )
            # set the start indices in fourth dimension
            start_data = 0
            start_label = 0

        if label_dataset is not None and frames_dataset is not None:
            # write the given data into the hdf5 file
            #reshaped_frames = np.transpose(frames, (3, 2, 0, 1))
            frames_dataset[start_data:start_data + frames.shape[0], :, :, :] = frames
            label_dataset[start_label:start_label + labels.shape[0], :] = labels

    finally:

        h5file.flush()
        h5file.close()


def reduce_dataset(frameSets):

    # only use frames that have labels
    def filter_relevant_frames(frameSet):
        frame_count = len(frameSet.frames)
        filtered_frames = [np.expand_dims(frameSet.frames[i], 0) for i in range(0, frame_count) if np.count_nonzero(frameSet.labels[i]) > 0]
        filtered_labels = [np.expand_dims(frameSet.labels[i], 0) for i in range(0, frame_count) if np.count_nonzero(frameSet.labels[i]) > 0]

        return frameSet.newStream(filtered_frames, frameSet.streamName, filtered_labels)

    return tuple(map(filter_relevant_frames, frameSets))


def post_process(frameSets):

    def resize(frameSet):
        resized_frames = map(lambda f: cv2.resize(f, (227, 227)), frameSet.frames)
        return frameSet.newStream(resized_frames, frameSet.streamName)


    def setMaskedToMean(frameSets):
        """Sets the black "masked" area aroudn the face to the mean value of the facial pixels

        Parameters
        ----------
        frameSets: Pair of Framesets

        Returns
        -------
        frameSets: Pair of FrameSets
            The input framesets with mean value in the masked area
        """

        def fillMaskedArea(frame, value):
            """Sets the black "masked" area aroudn the face to the mean value of the facial pixels

            Parameters
            ----------
            frame: X x Y Array
            value: the value to fill the masked area with

            Returns
            -------
            -
            """
            frame[frame == 0] = value
            return frame


        for frameSet in frameSets:
            meansPerFrame = calc_mean(frameSet)
            for frameWithLayers, means in zip(frameSet.frames, meansPerFrame):
                for layerI in range(0, len(frameWithLayers[0][0])):
                    frame = frameWithLayers[0:len(frameWithLayers), 0:len(frameWithLayers[0]), layerI]
                    mean = means[layerI]
                    fillMaskedArea(frame, mean)

    #setMaskedToMean(frameSets)
    return map(resize, frameSets)


def normalize_images(frameSets):

    def normalize_func(np_image):
        # normalize the image to contain values from 0 to 1 in each channel
        minval = np_image.min()
        maxval = np_image.max()
        if minval != maxval:
            np_image -= minval
            np_image *= (1.0 / (maxval-minval))
        return np_image

    def normalize(frameSet):
        resized_frames = map(normalize_func, frameSet.frames)
        return frameSet.newStream(resized_frames, frameSet.streamName)

    return map(normalize, frameSets)


def calc_mean(frameSet):
    def meansFrameSet(frameSet):
        """calculate the means over all frames and depth

        Parameters
        ----------
        frameSet: Frameset with component "frames" being #frames x X x Y x Z Array

        Returns
        -------
        meansOfLayers: Array
            #frames x Z Array containing the means

        """
        return [[meanLayer(frame[0:len(frame), 0:len(frame[0]), layerI]) for layerI in range(0, len(frame[0][0]))] for frame in frameSet.frames]



    def meanLayer(layer):
        """calculate the mean of a layer ignoring all zero elements

        Parameters
        ----------
        layer : X x Y Array

        Returns
        -------
        double
            mean value of the layer

        """
        return np.sum(layer) / np.count_nonzero(layer)

    return meansFrameSet(frameSet)


def get_all_videos(root_dir):

  # read the content of the root directory and filter all directories
  directory_names = map(lambda f: os.path.join(root_dir, f), os.listdir(root_dir))
  directories = filter(os.path.isdir, directory_names)

  filenames = []

  for directory in directories:
    for parent_dir, sub_dirs, files in os.walk(directory):

      # sort files
      for filename in natsorted(files):
        if filename.endswith("avi"):
            absolute_file = os.path.join(root_dir, parent_dir, filename)
            filenames.append(absolute_file)
  return natsorted(filenames)

def write_labels_to_disk(root_dir):

  # open ouput file
  filename = os.path.join(root_dir, "labelmapping.txt")
  output_file = open(filename, "w")

  output_file.write("Mapping FACS label -> integer label\n")
  for key in sorted(label_mapping, key=label_mapping.get):
    line = '{} {}\n'.format(key, label_mapping[key])
    output_file.write(line)

  # close output file
  output_file.close()

def main():
    if len(sys.argv) < 3:
        sys.exit("Usage: %s <path_to_video_directory> <output_path>" % sys.argv[0])

    # read path to image as command argument
    video_path = os.path.abspath(sys.argv[1])
    output_path = os.path.abspath(sys.argv[2])

    if not os.path.isdir(video_path):
        sys.exit("The specified <path_to_video_directory> argument is not a valid directory")

    if not os.path.isdir(output_path):
        sys.exit("The specified <output_path> argument is not a valid directory")


    flow_means = list() ; frame_means = list()

    allvideos = get_all_videos(video_path)

    print "About to process %d videos." % len(allvideos)

    for (i,video) in enumerate(allvideos):

        print "Processing video: %s" % video
        sys.stdout.write("\rProcess: %.1f%%\n" % (100.*i/len(allvideos)))
        sys.stdout.flush()

        # ready to rumble
        framesGray, framesBGR = read_video(video)

        for framesGray, framesBGR in itertools.izip(multiply_frames(framesGray), multiply_frames(framesBGR)):
            # 1. find faces 2. calc flow 3. save to disk
            frames = face_pass(framesGray, framesBGR)
            flows = flow_pass(frames[0])
            frames = post_process(frames)
            flows = post_process(flows)

            frames = normalize_images(frames)
            flows = normalize_images(flows)


            frames = reduce_dataset(frames)

            flows = reduce_dataset(flows)

            # save_to_disk(output_path, frames)
            # save_to_disk(output_path, flows)

            #print np.mean(flows[0].frames[0], axis=(0,1))[0]

            #flow_means += calc_mean(flows)
            #frame_means += calc_mean(frames)

            if random.random() > 0.9:
                postfix = "test"
            else:
                postfix = "train"

            save_as_hdf5(output_path, frames[1], "framesBGR_%s" % postfix)
            map(lambda flow: save_as_hdf5(output_path, flow, "flows_%s" % postfix), flows)


    #display means
    print flow_means
    print frame_means

    #write labels to disk
    write_labels_to_disk(output_path)

    # exit
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
