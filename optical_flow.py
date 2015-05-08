from __future__ import generators
import cv2, os, sys
import numpy as np

IMAGE_PATH = "/Users/therold/Dropbox/Uni/Practical Video Analyses/data/CK/cohn-kanade/S010/001/"
CLASSIFIER_PATH = "/Users/therold/Dropbox/Uni/Practical Video Analyses/haarcascade.xml"

faceCascade = cv2.CascadeClassifier(CLASSIFIER_PATH)


def list_files(dir):
    "walk a directory tree, using a generator"
    for f in sorted(os.listdir(dir)):
        fullpath = os.path.join(dir,f)
        yield fullpath

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


def face_pass(files):

    minX = minY = sys.maxint
    maxWidth = maxHeight = 0
    images = []

    for file in files:
        image = cv2.imread(file)
        (x, y, w, h) = detect_face(image)


        minX = min(minX, x)
        minY = min(minY, y)
        maxWidth = max(maxWidth, w)
        maxHeight = max(maxHeight, h)

        images.append(image)

    for image in images:
        yield image[minY : minY + maxHeight, minX : minX + maxWidth]


def flow_pass(images):


    prev = images.next()
    hsv = np.zeros_like(prev)
    hsv[..., 1] = 255

    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(prev)

    for image  in images:

        next = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(next)

        # prev, next, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow]
        flow = cv2.calcOpticalFlowFarneback(prev, next,  0.5,  3,  15,  3,  2,  1.1,  0)

        mag,  ang = cv2.cartToPolar(flow[..., 0],  flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame', prev)
        cv2.imshow('flow', rgb)

        # Wait indefinitely for user input
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', next)
            cv2.imwrite('opticalhsv.png', rgb )

        prev = next


    cv2.destroyAllWindows()


def main():


    image_files = list_files(IMAGE_PATH)
    images = face_pass(image_files)
    flow_pass(images)


if __name__ == "__main__":
    main()


