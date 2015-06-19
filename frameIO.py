from __future__ import generators
import cv2, os, sys, itertools, functools, h5py, random, numpy as np, xml.etree.ElementTree as ET
from natsort import natsorted
from frameset import *
from collections import defaultdict

NUMBER_OF_LABELS = 37

label_mapping = dict()
label_mapping_index = 0

def get_labels(video_file_name, num_frames):
    label_file = video_file_name.replace(".avi", "-oao_aucs.xml")
    return read_labels(label_file, num_frames)

def get_frames(video_file_name):
    # read video
    frames = []
    cap = cv2.VideoCapture(video_file_name)
    frame_count = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    if cap.isOpened():

        for i in range(0, frame_count):
            # actually read a frame
            returnValue, frame = cap.read()

            if not returnValue:
                break

            frames.append(frame)

        cap.release()

        return frames
        #return (FrameSet(framesGray, "frame-gray", "normal", labels), FrameSet(framesBGR, "frame-bgr", "normal", labels))
    else:
        sys.exit("Error opening video file.")

def save_to_disk_as_image(output_path, frameSets):
    for frameSet in frameSets:
        print "save_to_disk:", frameSet.processName, frameSet.streamName, frameSet.frames[0].shape
        for i, frame in enumerate(frameSet.frames):            
            if frameSet.streamName.startswith("flow"):
                for layer in range(0,len(frame[0][0])):
                    flatFrame = frame[:, :, layer]
                    cv2.imwrite(os.path.join(output_path, "%s_%s_%s_%s.png" % (frameSet.processName, frameSet.streamName, i, layer)), flatFrame)    
            else:
                cv2.imwrite(os.path.join(output_path, "%s_%s_%s.png" % (frameSet.processName, frameSet.streamName, i)), frame)

def transform_to_caffe_format(frameSets):
    """
    transforms the opencv format Frames x X x Y x Layers into caffe format Frames x Layers x X x Y
    """
    for frameSet in frameSets:
        swapped_axes = [np.swapaxes(np.swapaxes(frame, 0, 2), 1, 2) for frame in frameSet.frames]
        frames_as_big_blob = np.array(swapped_axes)
        yield frameSet.newFrames(frames_as_big_blob)

def save_as_hdf5_tree(output_path, db_name, frameSets):
    """
        saves frameSets as hdf5, expects the frames to be in caffe format Frames x Layers x X x Y as one big nparray
    """
    f = h5py.File(os.path.join(output_path, db_name), "w")
    
    for frameSet in frameSets:
        dataset_name = "/".join([frameSet.processName, frameSet.streamName])
        f.create_dataset(dataset_name + "/data", data = frameSet.frames, dtype="uint8")
        f.create_dataset(dataset_name + "/label", data = frameSet.labels, dtype="uint8")

    f.flush()
    f.close()

def read_from_hdf5_tree(hdf5_file):
    """
        reads frameSets from hdf5, frames are in caffe format Frames x Layers x X x Y as one big nparray
    """
    f = h5py.File(hdf5_file, "r")
    for processName, streams in f.items():
        for streamName, dataAndLabels in streams.items(): 
            yield FrameSet(dataAndLabels["data"].value, streamName, processName, dataAndLabels["label"].value)
    f.close()

def save_for_caffe(output_path, frameSets, DEBUG=False):
    def build_db_name(output_path, frameSet, filename_counters):
        db_prefix = None
        if frameSet.streamName == "BGR":
            db_prefix = "framesBGR_%s" % frameSet.getDbPostfix()
        elif frameSet.streamName.startswith("flow"):
            db_prefix = "flows_%s" % frameSet.getDbPostfix()
        else:
            raise NotImplementedError

        db = "%s_%d.h5" % (db_prefix, filename_counters[db_prefix])
        return db_prefix, os.path.join(output_path, db)

    def initialize_db(h5File, frameSet):
        max_shape_data = (None,) + frameSet.frames[0].shape
        h5File.create_dataset("data", maxshape=max_shape_data, data=frameSet.frames, chunks=True, dtype="uint8")

        max_shape_label = (None, ) + frameSet.labels[0].shape
        h5File.create_dataset("label", maxshape=max_shape_label, data=frameSet.labels, chunks=True, dtype="uint8")

    max_file_size = 1000 * 1000 * 1000 # 1 GB
    filename_counters = defaultdict(int)

    for frameSet in frameSets:
        db_prefix, db = build_db_name(output_path, frameSet, filename_counters)
        
        if os.path.isfile(db) and os.stat(db).st_size + frameSet.frames.nbytes > max_file_size:
            filename_counters[db_prefix] += 1
            db_prefix, db = build_db_name(output_path, frameSet, filename_counters)

        if DEBUG:
            print frameSet.processName, frameSet.streamName, db
        f = h5py.File(db)
        if "data" not in f:
            initialize_db(f, frameSet)
        else:
            data = f["data"]
            current_length = data.shape[0]
            new_length = current_length + frameSet.frames.shape[0]

            data.resize(new_length, axis=0)
            data[current_length:] = frameSet.frames

            label = f["label"]
            label.resize(new_length, axis=0)
            label[current_length:] = frameSet.labels

        f.flush()
        f.close()

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
  return filenames

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

def read_labels(path, num_frames):

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

    labels = np.zeros([num_frames, NUMBER_OF_LABELS])
    for au in action_units:

        facs_code = au.get("Number")

        for marker in au.findall("Marker"):
            frame_number = int(marker.get("Frame")) - 1

            label = map_label(facs_code)
            labels[frame_number, label] = 1

    return labels
