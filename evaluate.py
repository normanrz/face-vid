from __future__ import generators
import sys
import caffe
import matplotlib
import numpy as np
import h5py
import argparse
from collections import defaultdict
from frameIO import *
from frameset import *
from extract_frames import *


def read_means(means_file_name):
    with io.open(means_file_name, "r") as f:
        s = f.read()
        return json.loads(s)


def avg_net_results(probs):
    return reduce(lambda x, y: x + y, probs) / 2

def main():
    if len(sys.argv) < 3:
        sys.exit("Usage: %s <video_file> <means_file>" % sys.argv[0])

    video_file_name = sys.argv[1]
    means_file_name = os.path.abspath(sys.argv[2])

    frames = get_frames(video_file_name)
    means = read_means(means_file_name)
    frameSets = preprocess_frames(frames, means)

    prob_result = avg_net_results([
        forward_net(frameSets[1], "nets/an-finetune/deploy.prototxt", "nets/ANF_iter_50000.caffemodel"),
        forward_net(frameSets[1], "nets/an-finetune/deploy.prototxt", "nets/ANF_iter_50000.caffemodel")
    ])

    print prob_result
    label_result = prob_result > 0.5
    print label_result



def preprocess_frames(frames, means):

    processId = id_generator()
    face_cache = {}
    labels = np.zeros((1,1,1,1))

    # Extract frames
    frameSet = FrameSet(frames, "framesOriginal", processId, labels)

    frameSets = split_grayscale_BGR(frameSet)
    frameSets = detect_faces_and_mask_surroundings(frameSets, face_cache)
    frameSets = induce_flows(frameSets)
    frameSets = filter_framesets_out_by_stream_name(frameSets, "grayscale")
    # frameSets = filter_frames_with_labels(frameSets)
    frameSets = resize_frames(frameSets, 227, 227)
    # frameSets = accumulate_means(frameSets, means, layer_counts)
    frameSets = transform_to_caffe_format(frameSets)

    # Finalize (e.g. mean substraction)
    frameSets = substract_means(frameSets, means)
    frameSets = set_mask_to_zero(frameSets)
    frameSets = normalize_frames(frameSets)
    frameSets = list(mark_as_test(frameSets, 0.9))
    frameSets = cross_flows(frameSets)

    return list(frameSets)

def forward_net(frameSet, proto, model):
    net = caffe.Net(proto, model, caffe.TEST)
    # caffe.set_mode_gpu()

    print "Starting to forward data..."
    out = net.forward_all(data=frameSet.frames)
    print "Prediction finished"

    return out['prob']




if __name__ == "__main__":
    main()

