#!/usr/bin/env python3
""" computer vision AR demo"""

# ------- imports ------------------------------------------------------------
import sys
import logging
import argparse as ap
import cv2
import numpy as np
import time


# ------- argument parser ----------------------------------------------------
parser = ap.ArgumentParser(description='Simple augmented reality application.')
parser.add_argument('file', metavar='video_file', type=str, nargs='?',
                    help='a video file to apply AR on')
parser.add_argument('-o', '--output', type=str, nargs='?',
                    help='save live input stream as a video file')
parser.add_argument('-g', '--grid-size', type=int, nargs=2, default=(6, 5),
                    help='calibration grid size')
parser.add_argument('-l', '--log', choices=['NOTSET', 'DEBUG', 'INFO',
                                            'WARNING', 'ERROR', 'CRITICAL'],
                    help='Log level used to display logs', default='INFO')

args = parser.parse_args()

# logging
fmt = '[%(asctime)s.%(msecs)-3.2s%(levelname)s] %(message)s'
logging.basicConfig(level=getattr(logging, args.log),
                    format=fmt, datefmt='%H:%M:%S')
logging.info('Python version ' + sys.version)
logging.info('Arguments: ' + str(args))


# ------- draw augmentation object (cube) -------------------------------------
# function to draw our augmented scene / cube
def draw(image, points2d):
    # TODO 3: draw virtual object on real frame
    points2d = np.int32(points2d).reshape(-1, 2)

    # draw ground floor in green
    image = cv2.drawContours(image, [points2d[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        image = cv2.line(image, points2d[i], points2d[j], 255, 3)

    # draw top layer in red color
    image = cv2.drawContours(image, [points2d[4:]], -1, (0, 0, 255), 3)

    return image


# ------- main initialization and loop ----------------------------------------
def main():

    # get source video capture object
    # cam = cv2.VideoCapture(0 if args.file is None else args.file)
    cam = cv2.VideoCapture(0 if args.file is None else "videos/demo1.mp4")

    # define the codec & create VideoWriter object in case we write video output
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        dim = int(cam.get(3)), int(cam.get(4))  # float width and height
        out = cv2.VideoWriter(args.output, fourcc, 20.0, dim)

    # open window where we will show video stream
    cv2.namedWindow("Calibration and augmented reality demo")

    # construct array of checkerboard coordinates in object coordinate frame
    grid = args.grid_size
    object_points = [[i, j, 0] for j in range(grid[1]) for i in range(grid[0])]
    object_points = np.array(object_points, np.float32)
    print(f'{object_points=}')

    # Arrays to store object points and image points from all the images.
    points3d = []    # 3d points of calibration in its coordinate frame
    points2d = []    # matching 2d points in image plane.

    # variables that will contain calibration result
    intrinsics, distortion = None, None

    # initialize coordinates that will be used to draw the cube
    cube = np.float32([[0, 0, 0],  [0, 3, 0],  [3, 3, 0],  [3, 0, 0],
                       [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

    print('\n\nWave the calibration checkboard to the camera for a few frames, '
          'then press C to calibrate and see AR cube on checkerboard.\n')
    print('Press ESC to close.')

    # corner refinement termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    image_counter = 0
    scale = 0.3

    # main loop ---------------------------------------------------------------
    while cam.isOpened():
        # read an image and get return code
        ret, image = cam.read()

        # exit loop if we reach the end of stream
        if not ret:
            break

        # read key
        k = cv2.waitKey(1)

        # gray image conversion
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # examine key events
        if k & 0xFF == 27:
            # exit if ESC pressed
            logging.info("Escape hit, closing...")
            break
        elif k & 0xFF == ord('c') and intrinsics is None:
            # when user presses 'c', perform calibration
            # TODO 2.1: use the point3d <-> point2d matches to calibrate
            _, intrinsics, distortion, _, _ = cv2.calibrateCamera(points3d, points2d, gray.shape[::-1], None, None)

            logging.info('Calibration done')
            logging.info(f'Estimated distortion\n{distortion}')
            logging.info(f'Estimated intrinsics matrix\n{intrinsics}')

        if out:
            out.write(image)

        # TODO 1.1: chessboard detection on scaled down image for improved speed
        grays = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
        ret, corners = cv2.findChessboardCorners(grays, grid, None)

        if ret:
            points3d.append(object_points)

            # TODO 1.3: if corners found, get sub-pixel refinement of corners
            corners = corners / scale
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            points2d.append(corners.squeeze(axis=1))

            image_counter += 1
            if intrinsics is None:    # calibration ongoing

                # only take one in 30 frames for calibration to avoid overload
                if image_counter % 30 == 0:
                    # TODO 2.1: append each new point2d detection, and append
                    # the corresponding object point coordinates each time

                    rms, intrinsics, distortion, rvecs, tvecs = cv2.calibrateCamera(points3d, points2d, gray.shape[::-1], None, None)

                    # show frames used for calibration as inverted
                    image = cv2.bitwise_not(image)

                # TODO 1.2: draw and display the corners
                image = cv2.drawChessboardCorners(image, grid, corners, ret)

            else:
                # now that camera is calibrated, we can estimate extrinsics
                # TODO 2.2  find rotation and translation vectors.
                objpts = np.vstack(points3d) # shape: (N, 3)
                imgpts = np.vstack(points2d) # shape: (N, 2)
                ret, rotation_vectors, translation_vectors, _ = cv2.solvePnPRansac(objpts[-80:, :], imgpts[-80:, :], intrinsics, distortion)

                # TODO 2.3 project 3D points to image using estimated parameters
                projected_points, jacobian = cv2.projectPoints(cube, rotation_vectors, translation_vectors, intrinsics, distortion)

                image = draw(image, projected_points.squeeze(axis=1))

        cv2.imshow('img', image)
        time.sleep(0.1)

    # finalize and release stream objects, input and output when applicable
    cam.release()
    if out:
        out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
