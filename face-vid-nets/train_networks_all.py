#!/usr/bin/python

import os
import subprocess
import caffe
import numpy as np

NUM_LABELS=37

# defines the probability of 
LABEL_RATIOS = [0.2798239608801956, 0.032, 0.03589242053789731, 0.21116870415647923, 0.03765281173594132, 0.1493398533007335, 0.04312958435207824, 0.018034229828850855, 0.05046454767726161, 0.059559902200489, 0.026210268948655257, 0.026797066014669926, 0.03006356968215159, 0.024254278728606356, 0.015061124694376529, 0.030572127139364302, 0.0019559902200488996, 0.08205378973105135, 0.04485085574572127, 0.04267970660146699, 0.06930073349633252, 0.02380440097799511, 0.03168704156479218, 0.01867970660146699, 0.02982885085574572, 0.020440097799511003, 0.018288508557457214, 0.011657701711491443, 0.023452322738386308, 0.021339853300733496, 0.0037163814180929096, 0.028948655256723715, 0.021515892420537898, 0.02892909535452323, 0.012518337408312959, 0.016410757946210268, 0.0]

base_dir = "one-vs-all"

proto_template_file = "%s/train_val.prototxt" % base_dir
solver_template_file = "%s/solver.prototxt" % base_dir

with open(proto_template_file) as f:
  proto_template = f.read()

with open(solver_template_file) as f:
  solver_template = f.read()

out_dir = "one-vs-all/generated-nets"

def write_infogainH(i, subdir):
  label_ratio = LABEL_RATIOS[i]
  H = np.array([
    [label_ratio, 0], 
    [0, 1 - label_ratio]], dtype = 'f4')

  blob = caffe.io.array_to_blobproto( H.reshape( (1,1,2,2) ) )
  with open(subdir + '/infogainH.binaryproto', 'wb' ) as f :
      f.write( blob.SerializeToString() )

def write_filled_template(config, template, output):
  replaced_contents = template
  for key, value in config.iteritems():
    replaced_contents = replaced_contents.replace("<%s>" % key, value)

  outfile = open(output, "w")
  outfile.write(replaced_contents)
  outfile.flush()
  outfile.close()

def train_network(i, solver_path):
  subprocess.call(['./%s/train_network_one.sh' % base_dir, solver_path, str(i)])

for i in range(0, NUM_LABELS):
  print "About to train classifier for label %d" % i
  add_head = "top: \"head_labels\""
  add_tail = "top: \"tail_labels\""

  if i == 0:
    add_head = ""
    slices = "slice_point: 1 \n"
    silence_config = "bottom: \"tail_labels\"\n"
  elif i == NUM_LABELS-1:  
    add_tail = ""
    slices = "slice_point: %d \n" % i
    silence_config = "bottom: \"head_labels\"\n"
  else:
    slices = "slice_point: %d \n slice_point: %d" % (i, i+1)
    silence_config = "bottom: \"tail_labels\"\n bottom: \"head_labels\""

  subdir = "%s/%s/" % (out_dir, i)
  if not os.path.exists(subdir):
    os.makedirs(subdir)
  slice_config = """
  %s
  top: \"selected_label\"
  %s
  slice_param {
    axis: 1 
    %s
  }""" % (add_head, add_tail, slices)



  proto_path = os.path.abspath("%s/train_val.prototxt" % subdir)
  solver_path = os.path.abspath("%s/solver.prototxt" % subdir)

  write_filled_template({
    "SLICE_CONFIG" : slice_config,
    "SILENCE_CONFIG": silence_config,
    "PROTO_PATH" : os.path.abspath(subdir)
     }, proto_template, proto_path)
  
  write_filled_template({
    "NET_NUMBER" : str(i), 
    "PROTO_PATH" : os.path.abspath(subdir)
    } , solver_template, solver_path)

  write_infogainH(i, subdir)

  train_network(i, solver_path)
