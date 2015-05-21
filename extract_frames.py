###############################################################################
#
# Use this script the extract frames from the MMI Facial Expresssion DB.
# Every n-th frame will be extracted. Frames will be processed in the following
# manner:
#   - converted to grey-scale
#   - cropping to detected faces
#   - black oval mask around face
#   - save optical flow along x & y axis
#
# Usage: extract_frames.py <max_frame_count> <path_to_video> <output_path>
#
###############################################################################
from __future__ import generators
import cv2, os, sys, itertools
import numpy as np

CLASSIFIER_PATH = os.path.join(os.path.dirname(sys.argv[0]), "haarcascade_face.xml")
SCALE_FLOW = 10
faceCascade = cv2.CascadeClassifier(CLASSIFIER_PATH)

# Do face detection and return the first face
def detect_face(image):

	faces = faceCascade.detectMultiScale(
		image,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.cv.CV_HAAR_SCALE_IMAGE
	)

	# Only use first result
	return faces[0]

# Special processing relevant for the MMI facial dataset
def preprocessMMI(image):

	# turn into greyscale
	imageAsGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cv2.equalizeHist(imageAsGray)


	imageAsYCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB) #change the color image from BGR to YCrCb format
	channels = cv2.split(imageAsYCrCb) #split the image into channels
	channels[0] = cv2.equalizeHist(channels[0]) #equalize histogram on the 1st channel (Y)
	imageWithEqualizedHist = cv2.merge(channels) #merge 3 channels including the modified 1st channel into one image
	imageAsBGR = cv2.cvtColor(imageWithEqualizedHist, cv2.COLOR_YCR_CB2BGR) #change the color image from YCrCb to BGR format (to display image properly)

	return (imageAsGray,imageAsBGR)


def read_video(video, max_frame_count, frame_to_facs={}):

	# read video
	framesGray = []
	framesBGR = []
	cap = cv2.VideoCapture(video)
	frame_count = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	stride = frame_count / float(max_frame_count)

	# only grab & compute every x-th frame or known frames of interest
	relevant_frames = []
	if len(frame_to_facs) > 0:
		relevant_frames = [frame for frame in frame_to_facs]
	else:
		relevant_frames = [int(i) for i in np.arange(0, frame_count, stride)]

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

			(imageAsGray, imageAsBGR) = preprocessMMI(frame)
			framesGray.append(imageAsGray)
			framesBGR.append(imageAsBGR)

		cap.release()
		return (framesGray, framesBGR)
	else:
		sys.exit("Error opening video file.")


# Invoke face detection, find largest cropping window and apply elliptical mask
def face_pass(framesGray, framesBGR):
	def crop_and_mask(frame, minX, minY, maxWidth, maxHeight):
		cropped_frame = frame[minY : minY + maxHeight, minX : minX + maxWidth]

		center = (int(maxWidth  * 0.5), int(maxHeight * 0.5))
		axes = (int(maxWidth * 0.4), int(maxHeight * 0.5))

		mask = np.zeros_like(cropped_frame)
		cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)

		return np.bitwise_and(cropped_frame, mask)

	minX = minY = sys.maxint
	maxWidth = maxHeight = 0

	for frame in framesGray:

		# image = cv2.imread(file)
		(x, y, w, h) = detect_face(frame)

		minX = min(minX, x)
		minY = min(minY, y)
		maxWidth = max(maxWidth, w)
		maxHeight = max(maxHeight, h)

	return (
		map(lambda f: crop_and_mask(f, minX, minY, maxWidth, maxHeight), framesGray),
		map(lambda f: crop_and_mask(f, minX, minY, maxWidth, maxHeight), framesBGR)
	)

def calculateFlow(frame1, frame2):
	flow = cv2.calcOpticalFlowFarneback(frame1, frame2,  0.5,  3,  15,  3,  2,  1.1,  0)
	horz = cv2.convertScaleAbs(flow[..., 0], None, 128 / SCALE_FLOW, 128)
	vert = cv2.convertScaleAbs(flow[..., 1], None, 128 / SCALE_FLOW, 128)
	return horz, vert

# Calculate the optical flow along the x and y axis
# always compares with the first image of the series
def flow_pass_static(framesGray):
	#TODO: might not be the flow we want, comparing only the first image to all others
	first = framesGray[0]
	flows = [calculateFlow(first, f) for f in framesGray]
	return [list(t) for t in zip(*flows)]

# Calculate the optical flow along the x and y axis
# always compares with the previous image in the series
def flow_pass_continuous(framesGray):
	flows = [calculateFlow(f1, f2) for f1,f2 in zip(framesGray[0]+framesGray, framesGray)]
	return [list(t) for t in zip(*flows)]

def save_to_disk(output_path, frames, name):


   # if len(frame_to_facs) > 0:
   #      relevant_frames = [frame for frame in frame_to_facs]
   #  else:
   #      relevant_frames = [int(i) for i in np.arange(0, frame_count, stride)]


	for i,frame in enumerate(frames):
		cv2.imwrite(os.path.join(output_path, "%s_%s.png" % (name, i)), frame)

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
	framesGray, framesBGR = read_video(video_path, max_frame_count)

	# 1. find faces 2. calc flow 3. save to disk
	face_pass_result = face_pass(framesGray, framesBGR)
	if face_pass_result:
		croppedFramesGray, croppedFramesBGR = face_pass_result
		optical_flows = flow_pass_static(croppedFramesGray)
		framesHorizontalFlow, framesVerticalFlow = [list(t) for t in zip(*optical_flows)]
		save_to_disk(output_path, croppedFramesBGR, "frame-bgr")
		save_to_disk(output_path, croppedFramesGray, "frame-gray")
		save_to_disk(output_path, framesHorizontalFlow, "flow-x")
		save_to_disk(output_path, framesVerticalFlow, "flow-y")

	# exit
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()


