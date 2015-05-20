###############################################################################
#
# Use this script the extract frames from the MMI Facial Expresssion DB.
#
# Beware, this script only works for videos that have 'OAO FACS' metadata.
# If you only want to extract frames look at 'extract_frames.py'.
#
###############################################################################

from __future__ import generators
from extract_frames import *
from collections import defaultdict
from sortedcontainers import SortedDict
import xml.etree.ElementTree as ET
import os, cv2, sys
import numpy as np

# Override method of extract_frame.py
def read_video(video, metadata):

    # read video
    frames = []
    cap = cv2.VideoCapture(video)
    frame_count = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    # only grab & compute every x-th frame
    relevant_frames = [frame for frame in metadata]

    if cap.isOpened():

        for i in range(0, frame_count):

            if not i in relevant_frames:
                # skip video frame without decoding
                cap.grab()
                continue
            else:
                # actually read a frame
                returnValue, frame = cap.read()

            if not returnValue:
                break

            image = preprocessMMI(frame)
            frames.append(image)

        cap.release()
        return frames
    else:
        sys.exit("Error opening video file.")


def parseXML(path):

  # Read and parse
  tree = ET.parse(path)
  root = tree.getroot()

  action_units = root.findall(".//ActionUnit")

  result = SortedDict()
  for au in action_units:

    facs_code = int(au.get("Number"))

    for marker in au.findall("Marker"):
      frame_number = int(marker.get("Frame"))

      frames = result.get(frame_number, list())
      frames.append(facs_code)

      result[frame_number] = frames

  return result

# Override method of extract_frame.py
def save_to_disk(output_path, metadata, frames_flows_gen, ):

    i = 0

    for frame, flow_x, flow_y in frames_flows_gen:

      # name files according to their metadata
      frame_number = metadata.keys()[i]
      facs_units = metadata.values()[i]

      facs_string = "_".join(map(str,facs_units))
      filename_postfix = "%s_%s" % (frame_number, facs_string)

      cv2.imwrite(os.path.join(output_path, "frame-%s.png" % filename_postfix), frame)
      cv2.imwrite(os.path.join(output_path, "flow-x-%s.png" % filename_postfix), flow_x)
      cv2.imwrite(os.path.join(output_path, "flow-y-%s.png" % filename_postfix), flow_y)
      i += 1


def post_process_mmi(frames_flows_gen):

  # Resize image to 224x224 as used by Googlenet
  for images in frames_flows_gen:
    yield map(lambda x: cv2.resize(x, (224,224)), images)


def main():

  if len(sys.argv) < 3:
    sys.exit("Usage: %s <path_to_video> <output_path>" % sys.argv[0])

  # read path to image as command argument
  video_path = os.path.abspath(sys.argv[1])
  output_path = os.path.abspath(sys.argv[2])

  if not os.path.isfile(video_path):
      sys.exit("The specified <path_to_video> argument is not a valid filename")

  if not os.path.isdir(output_path):
      sys.exit("The specified <output_path> argument is not a valid directory")

  # read metadata xml
  metadata_file = video_path.replace(".avi", "-oao_aucs.xml")
  metadata = parseXML(metadata_file)

  # ready to rumble
  frames = read_video(video_path, metadata)

  # 1. find faces 2. calc flow 3. save to disk
  save_to_disk(output_path, metadata, post_process_mmi(flow_pass(face_pass(frames))))

  # exit
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()


