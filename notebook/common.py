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

def column_centroids(image, mask=None):
	# range(image.cols) in column-vector form
	rowIdx = np.arange(image.shape[0]).reshape(-1,1)
	# an image with each row = rowIdx
	centroid_mask = np.hstack((rowIdx,)*image.shape[1])

	# Apply a mask to the image
	masked = np.zeros_like(image)
	masked[mask] = image[mask]

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