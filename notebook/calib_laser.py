#%%
from calib import calibrate_charuco
from utils import load_coefficients, save_coefficients
import cv2

import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import pathlib

plt.rcParams["figure.figsize"] = (16,9)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['image.cmap'] = 'gray'


# def calibrate_charuco(dirpath, image_format, marker_length, square_length, prior=None, plot=False):
dirpath = r'Imagens\setup0\filter'
image_format = 'jpg'
# Dimensions in cm
marker_length = 2.1
square_length = 3.5
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
board = aruco.CharucoBoard_create(5, 7, square_length, marker_length, aruco_dict)
arucoParams = aruco.DetectorParameters_create()
#%%

mtx, dist = load_coefficients('calibration_charuco.yml')

counter, corners_list, id_list = [], [], []
imgs = []
img_dir = pathlib.Path(dirpath)
first = 0
# Find the ArUco markers inside each image
for img in img_dir.glob(f'*{image_format}'):
	# print(f'using image {img}')
	image = cv2.imread(str(img))
	img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	img_gray = cv2.undistort(img_gray, mtx, dist, None, mtx)

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
	if resp > 20:
	# print(charuco_corners)
	# print(resp)
	# Add these corners and ids to our calibration arrays
		corners_list.append(charuco_corners)
		id_list.append(charuco_ids)
		imgs.append(img_gray)
	# else:
	# print("Failed!")

for points, img, ids  in zip(corners_list, imgs, id_list):

	ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(points, ids, board, mtx, dist, None, None)
	proj_points, jacobian = cv2.projectPoints(board.chessboardCorners, rvec, tvec, mtx, dist)

	# Defining the plane
	n_chessboard = np.array([0, 0, 1])
	T_to_plane = np.eye(4)
	M, jac = cv2.Rodrigues(rvec)
	n3 = M@n_chessboard
	p = -1 * n3@tvec
	n = np.hstack((n3,p))

	# rvec, tvec @ n_chessboard -> Plano em 3d do chessboard
	# interseccao raio-plano com cam_mtx e n_chessboard -> Pontos 3d do laser
	# (da pra fazer minimizacao tbm)
		
	# Pontos 3d do laser em multiplos frames -> plano do laser

	# print(projected)
	plt.imshow(img)
	plt.plot([p[0][0] for p in points], [p[0][1] for p in points], "r.", markersize=12)
	plt.plot([p[0][0] for p in proj_points], [p[0][1] for p in proj_points], "g.", markersize=12)
	plt.show()
# cv2.projectPoints(points, rvecs, tvecs, mtx, dist)