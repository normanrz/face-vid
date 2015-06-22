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
from frameIO import *
from frameset import *
import uuid
from collections import defaultdict
import json
import io
import string
import random


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
   return ''.join(random.choice(chars) for _ in range(size))

CLASSIFIER_PATH = os.path.join(os.path.dirname(sys.argv[0]), "haarcascade_frontalface_alt.xml")
SCALE_FLOW = 10
DEBUG = False

faceCascade = cv2.CascadeClassifier(CLASSIFIER_PATH)

# Special processing relevant for the MMI facial dataset
def split_grayscale_BGR(frameset):
    def split_frame_channels(frame):
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(frame_grayscale)

        frameAsYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)  # change the color image from BGR to YCrCb format
        channels = cv2.split(frameAsYCrCb)  # split the image into channels
        channels[0] = cv2.equalizeHist(channels[0])  # equalize histogram on the 1st channel (Y)
        frameWithEqualizedHist = cv2.merge(channels)  # merge 3 channels including the modified 1st channel into one image
        frame_BGR = cv2.cvtColor(frameWithEqualizedHist,
                                  cv2.COLOR_YCR_CB2BGR) # change the color image from YCrCb to BGR format (to display image properly)
        return (frame_grayscale, frame_BGR)

    frames_grayscale, frames_BGR = zip(*[split_frame_channels(frame) for frame in frameset.frames])

    yield FrameSet(np.expand_dims(frames_grayscale,3), "grayscale", frameset.processName, frameset.labels)
    yield FrameSet(frames_BGR, "BGR", frameset.processName, frameset.labels)

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

# Invoke face detection, find largest cropping window and apply elliptical mask
def detect_faces_and_mask_surroundings(frameSets, face_cache):
    def crop_and_mask(frame, minX, minY, maxWidth, maxHeight, count):
        cropped_frame = frame[minY: minY + maxHeight, minX: minX + maxWidth]

        center = (int(maxWidth * 0.5), int(maxHeight * 0.5))
        axes = (int(maxWidth * 0.4), int(maxHeight * 0.5))

        mask = np.zeros_like(cropped_frame)
        cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)

        return np.bitwise_and(cropped_frame, mask)

    def apply_mask(frameSet, mask):
        return frameSet.map(lambda f: crop_and_mask(f, **mask))

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

    for frameSet in frameSets:
        if DEBUG:
            print "detect_faces_and_mask_surroundings:", frameSet.processName, frameSet.streamName, frameSet.frames[0].shape
        if frameSet.streamName == "grayscale":
            known_faces = []
            for i, frame in enumerate(frameSet.frames):

                # only do face detection every 10 frames to save processing power
                if i % 10 <> 0:
                    continue

                # Find all faces
                for face in detect_face(frame):
                    known_faces = remember_face(face, known_faces)


            most_significant_face = max(known_faces, key=lambda x: x["count"])
            face_cache[frameSet.processName] = most_significant_face
            yield apply_mask(frameSet, most_significant_face)
        else:
            most_significant_face = face_cache.get(frameSet.processName, None)
            if most_significant_face:
                yield apply_mask(frameSet, most_significant_face)
            else:
                sys.exit("Fatal error: didn't find entry in facecache for frameset(%s):%s" % (frameSet.streamName, frameSet.processName))



def calculateFlow(frame1, frame2):
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, 0.5, 3, 15, 3, 2, 1.1, 0)
    horz = cv2.convertScaleAbs(flow[..., 0], None, 128 / SCALE_FLOW, 128)
    vert = cv2.convertScaleAbs(flow[..., 1], None, 128 / SCALE_FLOW, 128)
    return horz, vert


# Increase the dataset by adding some rotated and re-colored images
def multiply_frames(frameSets):
    def rotate_frame(frame, angle):
        rows = frame.shape[0]
        cols = frame.shape[1]
        rotMat = cv2.getRotationMatrix2D((rows / 2, cols / 2), angle, 1)
        return cv2.warpAffine(frame, rotMat, (cols, rows))

    def contrast_brightness(frame, gain, bias):
        return np.array(np.fmax(np.fmin(gain * np.array(frame, dtype=np.float32) + bias, 255), 0), dtype=np.uint8)

    for frameSet in frameSets:
        yield frameSet
        for i, angle in enumerate([0, -3, 3, -6, 6]):
            rotatedFrames = [rotate_frame(frame, angle) for frame in frameSet.frames]
            for j, gain in enumerate([0]):
                for k, bias in enumerate([-40, -20, 0, 20]):
                    newFrames = [contrast_brightness(frame, 1.1 ** gain, bias) for frame in rotatedFrames]
                    newProcessName = frameSet.processName + "-multi%i-%i-%i" % (i, j, k)
                    yield frameSet.newProcess(newFrames, newProcessName)


# Calculate the optical flow along the x and y axis
# always compares with the previous/next #x images in the series
def induce_flows(frameSets):

    sliding_window = 5

    # Find the 10 frames around each flow image and stack them as a single 10channel blob
    def stack_frames(flow_data, frame_number):

        relevant_frames = range(frame_number - sliding_window, frame_number + sliding_window)
        frames = map(lambda i: empty_frame if i < 0 or i >= len(flow_data) else flow_data[i], relevant_frames)
        return cv2.merge(frames)

    for frameSet in frameSets:
        if DEBUG:
            print "induce_flows..:", frameSet.processName, frameSet.streamName, frameSet.frames[0].shape
        if frameSet.streamName == "grayscale":
            flows = zip(*[calculateFlow(f1, f2) for f1, f2 in zip(frameSet.frames[0] + frameSet.frames, frameSet.frames)])
            empty_frame = np.ones_like(flows[0][0]) * 128

            stacked_flows  = map(lambda flow:
                map(functools.partial(stack_frames, flow), range(0, len(flow)))
            , flows)

            yield frameSet
            yield frameSet.newStream(stacked_flows[0], "flow-x")
            yield frameSet.newStream(stacked_flows[1], "flow-y")
        else:
            yield frameSet

def filter_framesets_out_by_stream_name(frameSets, stream_name):
    for frameSet in frameSets:
        if DEBUG:
            print "filter_framesets_out..:", frameSet.processName, frameSet.streamName, frameSet.frames[0].shape
        if frameSet.streamName != stream_name:
            yield frameSet

def filter_frames_with_labels(frameSets):
    for frameSet in frameSets:
        if DEBUG:
            print "filter frames with labels:", frameSet.processName, frameSet.streamName, frameSet.frames[0].shape
        frame_count = len(frameSet.frames)
        filtered_frames = [frameSet.frames[i] for i in range(0, frame_count) if np.count_nonzero(frameSet.labels[i]) > 0]
        filtered_labels = [frameSet.labels[i] for i in range(0, frame_count) if np.count_nonzero(frameSet.labels[i]) > 0]
        yield frameSet.newStream(filtered_frames, frameSet.streamName, filtered_labels)

def resizeFrames(frameSets, x, y):
    for frameSet in frameSets:
        if DEBUG:
            print "resizeFrames:", frameSet.processName, frameSet.streamName, frameSet.frames[0].shape
        resized_frames = map(lambda frame: cv2.resize(frame, (x, y)), frameSet.frames)
        if len(resized_frames[0].shape) < 3:
            resized_frames = map(lambda frame: np.expand_dims(frame, 2), resized_frames)
        yield frameSet.newStream(resized_frames, frameSet.streamName)

def accumulate_means(frameSets, means, layer_counts):
    for frameSet in frameSets:
        if DEBUG:
            print "accumulate_means:", frameSet.processName, frameSet.streamName, frameSet.frames[0].shape
        for frame in frameSet.frames:
            for layer in range(0, len(frame[0][0])):
                flatFrame = frame[:, :, layer]
                if frameSet.streamName.startswith("flow"):
                    means[frameSet.streamName][0] += np.sum(flatFrame)
                    layer_counts[frameSet.streamName][0] += np.count_nonzero(flatFrame)
                else:
                    means[frameSet.streamName][layer] += np.sum(flatFrame)
                    layer_counts[frameSet.streamName][layer] += np.count_nonzero(flatFrame)
        yield frameSet

def calculate_means(means, layer_counts):
    if DEBUG:
        print json.dumps(means)
        print json.dumps(layer_counts)
    for streamName, counts_per_layer in layer_counts.items():
        for layer, count in counts_per_layer.items():
            means[streamName][layer] = means[streamName][layer] / count
    if DEBUG:
        print json.dumps(means)
    return means

def set_masks_to_mean(frameSets, means):
    for frameSet in frameSets:
        for layer in range(0,len(frameSet.frames[0])):
            frameSet.frames[:, layer][frameSet.frames[:, layer] == 0] = means[frameSet.streamName][str(layer)]
        yield frameSet

def mark_as_test(frameSets, percentageTrainingSet):
    cache = {}
    for frameSet in frameSets:
        if frameSet.processName in cache:
            frameSet.markAsTest(cache[frameSet.processName])
        elif random.random() > 0.9:
            frameSet.markAsTest(True)
            cache[frameSet.processName] = True
        else:
            frameSet.markAsTest(False)
            cache[frameSet.processName] = False
        yield frameSet

def write_means(output_path, means):
    with io.open(os.path.join(output_path,"means"), "w") as f:
        f.write(unicode(json.dumps(means)))

def extraction_flow(video_path, output_path):
    intermediate_h5_file = "intermediate.h5"
    means = defaultdict(lambda: defaultdict(float))
    layer_counts = defaultdict(lambda: defaultdict(int))
    def extract_frames():
        face_cache = {}
        video_file_names = get_all_videos(video_path)
        for i, video_file_name in enumerate(video_file_names):
            processId = os.path.split(video_file_name)[1] + "-" + id_generator()
            print "Processing video: <%s> (%d/%d)" % (video_file_name, i, len(video_file_names))

            frames = get_frames(video_file_name)
            labels = get_labels(video_file_name, len(frames))
            frameSet = FrameSet(frames, "framesOriginal", processId, labels)

            frameSets = split_grayscale_BGR(frameSet)
            frameSets = multiply_frames(frameSets)
            frameSets = detect_faces_and_mask_surroundings(frameSets, face_cache)
            frameSets = induce_flows(frameSets)
            frameSets = filter_framesets_out_by_stream_name(frameSets, "grayscale")
            frameSets = filter_frames_with_labels(frameSets)
            frameSets = resizeFrames(frameSets, 227, 227)
            frameSets = accumulate_means(frameSets, means, layer_counts)
            frameSets = transform_to_caffe_format(frameSets)
            save_as_hdf5_tree(output_path, intermediate_h5_file, frameSets)

    def finalize():
        frameSets = read_from_hdf5_tree(os.path.join(output_path, intermediate_h5_file))
        frameSets = set_masks_to_mean(frameSets,means)
        frameSets = mark_as_test(frameSets, 0.9)
        save_for_caffe(output_path, frameSets)

    extract_frames()
    means = calculate_means(means, layer_counts)
    write_means(output_path, means)
    finalize()
    write_labels_to_disk(output_path)

    # exit
    cv2.destroyAllWindows()


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

    extraction_flow(video_path, output_path)
    


if __name__ == "__main__":
    main()
