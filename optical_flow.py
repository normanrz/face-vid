from __future__ import generators
import cv2, os, sys, math
import numpy as np

# CLASSIFIER_PATH = "/usr/local/Cellar/opencv/2.4.11/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
CLASSIFIER_PATH = os.path.join(os.path.dirname(sys.argv[0]), "haarcascade_face.xml")
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
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(img)

    # crop the image to 360x576
    return img[0:576, 360:720]


def read_video(video, max_frame_count):

    # read video
    frames = []
    cap = cv2.VideoCapture(video)
    frame_count = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    # only grab & compute every x-th frame
    stride = frame_count / float(max_frame_count)
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

            image = preprocessMMI(frame)
            frames.append(image)

        cap.release()
        return frames
    else:
        sys.exit("Error opening video file.")


# Invoke face detection, find largest cropping window and apply elliptical mask
def face_pass(images):

    minX = minY = sys.maxint
    maxWidth = maxHeight = 0

    for image in images:

        # image = cv2.imread(file)
        (x, y, w, h) = detect_face(image)

        minX = min(minX, x)
        minY = min(minY, y)
        maxWidth = max(maxWidth, w)
        maxHeight = max(maxHeight, h)


    for image in images:

        # crop image image to recognized face and apply elliptical mask
        cropped_image = image[minY : minY + maxHeight, minX : minX + maxWidth]

        center = (int(maxWidth  * 0.5), int(maxHeight * 0.5))
        axes = (int(maxWidth * 0.4), int(maxHeight * 0.5))

        mask = np.zeros_like(cropped_image)
        cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)

        yield np.bitwise_and(cropped_image, mask)


# Calculate the optical flow along the x and y axis
def flow_pass(images):

    prev = images.next()

    for next in images:

        # prev, next, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow]
        flow = cv2.calcOpticalFlowFarneback(prev, next,  0.5,  3,  15,  3,  2,  1.1,  0)

        # horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
        # vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
        horz = flow[..., 0]
        vert = flow[..., 1]

        horz = horz.astype("uint8")
        vert = vert.astype("uint8")

        yield horz, vert


def save_to_disk(output_path, frames, flow):

    i = 0
    for frame, flow in zip(frames, flow):

        cv2.imwrite(os.path.join(output_path, "frame_%s.jpg" % i), frame)
        cv2.imwrite(os.path.join(output_path, "flow_x_%s.jpg" % i), flow[0])
        cv2.imwrite(os.path.join(output_path, "flow_y_%s.jpg" % i), flow[1])

        i += 1


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
    frames = read_video(video_path, max_frame_count)

    frames = face_pass(frames)
    flow = flow_pass(frames)

    save_to_disk(output_path, frames, flow)

    # exit
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


