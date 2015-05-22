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


def parseXML(path):
    # Read and parse
    tree = ET.parse(path)
    root = tree.getroot()

    action_units = root.findall(".//ActionUnit")

    result = SortedDict()
    for au in action_units:

        facs_code = au.get("Number")

        for marker in au.findall("Marker"):
            frame_number = int(marker.get("Frame"))

            frames = result.get(frame_number, list())
            frames.append(facs_code)

            result[frame_number] = frames

    return result


# Override method of extract_frame.py
def save_to_disk_with_facs(output_path, frames, name, metadata):
    # only grab frames with FACS labels
    relevant_frames = [i for i in metadata.keys()]

    for i, frame in enumerate(frames):
        if not i in relevant_frames: continue

        # name files according to their metadata
        frame_number = i
        facs_units = metadata.get(i)
        facs_string = "_".join(map(str, facs_units))

        post_processed_frame = post_process_mmi(frame)
        cv2.imwrite(os.path.join(output_path, "%s-%s_%s.png" % (name, frame_number, facs_string)), post_processed_frame)


def post_process_mmi(frame):
    return cv2.resize(frame, (230, 230))


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
    framesGray, framesBGR = read_video(video_path)

    face_pass_result = face_pass(framesGray, framesBGR)
    if face_pass_result:
        croppedFramesGray, croppedFramesBGR = face_pass_result
        static_flows = flow_pass_static(croppedFramesGray)
        continous_flows = flow_pass_continuous(croppedFramesGray)

        save_to_disk_with_facs(output_path, croppedFramesBGR, "frame-bgr", metadata)
        save_to_disk_with_facs(output_path, croppedFramesGray, "frame-gray", metadata)
        save_to_disk_with_facs(output_path, static_flows[0], "static-flow-x", metadata)
        save_to_disk_with_facs(output_path, static_flows[1], "static-flow-y", metadata)
        save_to_disk_with_facs(output_path, continous_flows[0], "cont-flow-x", metadata)
        save_to_disk_with_facs(output_path, continous_flows[1], "cont-flow-y", metadata)

    # exit
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
