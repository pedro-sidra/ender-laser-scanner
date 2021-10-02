import numpy as np
import pickle
import cv2.aruco as aruco
import pathlib
import cv2
from tqdm.auto import tqdm as progressbar
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate(img_gen, interval=500):
	fig = plt.figure()
	ims = []
	for image in img_gen:
		image = plt.imshow(image, animated=True)
		ims.append([image])

	ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True,
									repeat_delay=1000)
	plt.show()
	return ani

def column_argmaxes(image, mask=None):
	# Apply a mask to the image
	masked = np.zeros_like(image)
	masked[mask] = image[mask]

	# Centroid of each line in image
	x = np.argmax(masked, axis=0).astype(np.float)
	x[x==0] = np.nan
	return x

def column_centroids(image, mask=None):
	# range(image.cols) in column-vector form
	rowIdx = np.arange(image.shape[0]).reshape(-1,1)
	# an image with each row = rowIdx
	centroid_mask = np.hstack((rowIdx,)*image.shape[1])

	# Apply a mask to the image
	masked = np.zeros_like(image)
	masked[mask] = image[mask]
	
	masked = masked.astype(np.float)

	# Centroid of each line in image
	a = np.sum(masked * centroid_mask, axis=0)
	b = np.sum(masked, axis=0)
	centroids = np.empty_like(a)
	centroids[:] = np.nan
	# Divide but not where b==0
	centroids = np.divide(a, b, out = centroids, where=b!=0)

	return centroids

def red_contrast(image):
	"""Calculate red - (green+blue)/2 for an rgb image. Return in float"""
	image = image.astype(np.float)
	return image[:,:, 0] - np.mean(image[:,:,1:], axis=-1)

def decode(r, rgb=True):
	"""Decode requestslib response r as an image

	@param r response object from `requests` lib
	@param rgb use rgb instead of opencv's bgr
	"""
	im = cv2.imdecode(
		np.asarray(bytearray(b"".join(r)),
				dtype=np.uint8),
		flags=-1)

	if rgb:
		return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	else:
		return im

def hdr(images, times, gamma=2.2):
    # Merge exposures to HDR image
    merge_debevec = cv2.createMergeDebevec()
    hdr_debevec = merge_debevec.process(images, times=np.array(times, dtype=np.float32))

    tonemap1 = cv2.createTonemap(gamma=gamma)
    res_debevec = tonemap1.process(hdr_debevec.copy())
    res_debevec = np.clip(res_debevec*255, 0, 255).astype('uint8')
    return res_debevec

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_axes(ax, M, scale=10):
    # X
    tvec = M[:3,3]
    R = M[:3,:3]
    R*=scale
    ax.quiver(tvec[0],tvec[1], tvec[2], R[0,0], R[1,0], R[2,0], color="r")
    # Y
    ax.quiver(tvec[0],tvec[1], tvec[2], R[0,1], R[1,1], R[2,1], color="g")
    # Z
    ax.quiver(tvec[0],tvec[1], tvec[2], R[0,2], R[1,2], R[2,2], color="b")

def vec2M(rvec, tvec):
	R, jac = cv2.Rodrigues(rvec)
	inv = np.zeros((4,4))
	inv[:3, :3] = R
	inv[:3, 3] = tvec.T
	inv[3,3] = 1

	return inv

def plot3d(**kwargs):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d', **kwargs)
	# Chessboard
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("z")
	return fig, ax