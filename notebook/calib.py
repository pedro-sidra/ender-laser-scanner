import numpy as np
import cv2
import pickle
import cv2.aruco as aruco
# import matplotlib
# import matplotlib.pyplot as plt
import pathlib

# matplotlib.use("Qt5Agg")

# plt.rcParams['figure.constrained_layout.use'] = True
# plt.rcParams['image.cmap'] = 'gray'

def save_board(filename, board, dict_name):
    """Saves aruco board (needs dict name)
    @param filename file to save
    @param board `cv2.arucoboard` to save
    @param dict_name id for `cv2.aruco.Dictionary_create`"""
    out = dict( 
        chessboard_size=board.getChessboardSize(),
        square_length = board.getSquareLength(),
        marker_length=board.getMarkerLength(),
        chessb_id = dict_name
    )
    with open(filename, "wb") as f:
        pickle.dump(out, f)

def load_board(filename):
    """Loads aruco board from file
    @param filename file to load
    """
    with open(filename, "rb") as f:
        params = pickle.load(f)
    aruco_dict = aruco.Dictionary_get(params["chessb_id"])
    return (aruco.CharucoBoard_create(*params["chessboard_size"], params["square_length"], params["marker_length"], aruco_dict),
            aruco_dict)

def calibrate_charuco(dirpath, image_format, marker_length, square_length, prior=None, plot=False):
    '''Apply camera calibration using aruco.
    The dimensions are in cm.
    '''
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
    board = aruco.CharucoBoard_create(8, 8, square_length, marker_length, aruco_dict)
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
        
        print(img)
        # cv2.imshow("TEST", img_gray)
        # cv2.waitKey(-1)

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

def calibrate_charuco_local(image_gen, board, aruco_dict, prior=None, plot=False):
    arucoParams = aruco.DetectorParameters_create()

    counter, corners_list, id_list = [], [], []
    imgs = []
    idxs=[]
    first = 0
    # Find the ArUco markers inside each image
    for i, image in enumerate(image_gen):
        # print(f'using image {img}')
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if prior is not None:
            mtx_pr, dist_pr = prior
        #     img_gray = cv2.undistort(img_gray, mtx_pr, dist_pr, None, mtx_pr)

        corners, ids, rejected = aruco.detectMarkers(
            img_gray, 
            aruco_dict, 
            parameters=arucoParams,
            cameraMatrix=mtx_pr if prior else cv2.noArray(),
            distCoeff=dist_pr if prior else cv2.noArray(),
        )

        resp, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=img_gray,
            board=board,
            cameraMatrix=mtx_pr if prior else cv2.noArray(),
            distCoeffs=dist_pr if prior else cv2.noArray(),
        )
        if resp > 6:
            # Add these corners and ids to our calibration arrays
            corners_list.append(charuco_corners)
            id_list.append(charuco_ids)
            imgs.append(img_gray)
            idxs.append(i)
        else:
            print(f"skipped image {i}, had only {resp} markers")

    # Actual calibration
    if prior is not None:
        mtx_pr, dist_pr = prior
        # img_gray = cv2.undistort(img_gray, mtx_pr, dist_pr, None, mtx_pr)

    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_list, 
        charucoIds=id_list, 
        board=board, 
        imageSize=img_gray.shape, 
        cameraMatrix=mtx_pr if prior else None, 
        distCoeffs=dist_pr if prior else None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS if prior else None)

    for points, img, rvec, tvec in zip(corners_list, imgs, rvecs, tvecs):
        proj_points, jacobian = cv2.projectPoints(board.chessboardCorners, rvec, tvec, mtx, dist)
        # print(projected)
        if plot:
            plt.figure()
            plt.imshow(img)
            plt.plot([p[0][0] for p in points], [p[0][1] for p in points], "r.", markersize=12)
            plt.plot([p[0][0] for p in proj_points], [p[0][1] for p in proj_points], "g.", markersize=12)
            plt.show()
    # cv2.projectPoints(points, rvecs, tvecs, mtx, dist)

    print(f"reprojection error: {ret}")
    
    return [idxs, mtx, dist, rvecs, tvecs, corners_list]

def save_coefficients(mtx, dist, path):
    '''Save the camera matrix and the distortion coefficients to given path/file.'''
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

def load_coefficients(path):
    '''Loads camera matrix and distortion coefficients.'''
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]