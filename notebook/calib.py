import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import pathlib

plt.rcParams["figure.figsize"] = (16,9)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['image.cmap'] = 'gray'


def calibrate_charuco(dirpath, image_format, marker_length, square_length, prior=None, plot=False):
    '''Apply camera calibration using aruco.
    The dimensions are in cm.
    '''
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
    board = aruco.CharucoBoard_create(5, 7, square_length, marker_length, aruco_dict)
    arucoParams = aruco.DetectorParameters_create()

    counter, corners_list, id_list = [], [], []
    imgs = []
    img_dir = pathlib.Path(dirpath)
    first = 0
    # Find the ArUco markers inside each image
    for img in img_dir.glob(f'*{image_format}'):
        # print(f'using image {img}')
        image = cv2.imread(str(img))
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if prior is not None:
            mtx_pr, dist_pr = prior
            img_gray = cv2.undistort(img_gray, mtx_pr, dist_pr, None, mtx_pr)

        corners, ids, rejected = aruco.detectMarkers(
            img_gray, 
            aruco_dict, 
            parameters=arucoParams
        )

        resp, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=img_gray,
            board=board
        )
        # If a Charuco board was found, let's collect image/corner points
        # Requiring at least 20 squares
        if resp > 10:
            # print(charuco_corners)
            # print(resp)
            # Add these corners and ids to our calibration arrays
            corners_list.append(charuco_corners)
            id_list.append(charuco_ids)
            imgs.append(img_gray)
        # else:
            # print("Failed!")

    # Actual calibration
    if prior is not None:
        mtx_pr, dist_pr = prior
        img_gray = cv2.undistort(img_gray, mtx_pr, dist_pr, None, mtx_pr)

    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_list, 
        charucoIds=id_list, 
        board=board, 
        imageSize=img_gray.shape, 
        cameraMatrix=mtx_pr if prior else None, 
        distCoeffs=None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS if prior else None)

    for points, img, rvec, tvec in zip(corners_list, imgs, rvecs, tvecs):
        proj_points, jacobian = cv2.projectPoints(board.chessboardCorners, rvec, tvec, mtx, dist)
        # print(projected)
        if plot:
            plt.imshow(img)
            plt.plot([p[0][0] for p in points], [p[0][1] for p in points], "r.", markersize=12)
            plt.plot([p[0][0] for p in proj_points], [p[0][1] for p in proj_points], "g.", markersize=12)
            plt.show()
    # cv2.projectPoints(points, rvecs, tvecs, mtx, dist)
    
    return [ret, mtx, dist, rvecs, tvecs]