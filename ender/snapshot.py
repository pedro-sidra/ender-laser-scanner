import requests
from threading import Thread
from collections import deque
import subprocess
import time
import shutil
import numpy as np
import cv2


class MjpgSnapshotter():
	def __init__(self, url, snapshot_params={"action":"snapshot"}, rgb=True) -> None:
	    self.url = url
	    self.params = snapshot_params
	    self.rgb=rgb
	
	def get_raw(self):
		r = requests.get(self.url, params=self.params, stream=True)
		if r.status_code==200:
			return r
		else:
			raise Exception("Response not ok!", r.status_code)

	def get_img(self):
		im = cv2.imdecode(
			np.asarray(bytearray(b"".join(self.get_raw())),
				   dtype=np.uint8),
			flags=-1)
		
		if self.rgb:
			return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		else:
			return im
	def read(self):
		try:
			return True, self.get_img()
		except:
			return False, None

class MjpgSnapshotterExp(MjpgSnapshotter):
	"""Extends `MjpgSnapshotter` by adding v4l2 commands to the host device

	Can adjust exposure via `set_exposure`, gain via `set_gain`
	and others via `send_command`, to get available commands call `get_commands`
	"""
	def __init__(self, url,
				 snapshot_params={"action":"snapshot"}, 
				 rgb=True,
				 ssh_str=None) -> None:
		"""Init a virtual camera that GETs frames from `url` with `snapshot_params`,
		and can send camera parameter commands by SSH connection using `ssh_str`

		Always uses the first available camera from v4l2-ctl
		:param url:url hosting the webcam API
		@snapshot_params params to use at the API to the webcam
		:param rgb:whether to convert the image from opencv's BGR to matplotlib's RGB
		@ssh_str acess str to use for ssh e.g. pi@ender3.local
		"""
		super().__init__(url, snapshot_params=snapshot_params, rgb=rgb)
		self.ssh_str = ssh_str
		# Turn off auto-exp
		self._ssh_command("v4l2-ctl -cexposure_auto=1")
		# Turn off white balance
		self._ssh_command("v4l2-ctl -cwhite_balance_temperature_auto=0")
		self._ssh_command("v4l2-ctl -cwhite_balance_temperature=0")

	def _ssh_command(self, cmd):
		"""Send ssh command to host"""

		if self.ssh_str is None:
			command = cmd.split()
		else:
			command = [ "ssh", self.ssh_str, cmd ]

		try:
			return subprocess.check_output(command).decode()
		except subprocess.CalledProcessError:
			time.sleep(0.5)
			return subprocess.check_output(command).decode()
	
	def set_exposure(self, value, wait=True):
		"""Set the exposure using -cexposure=value
		:param wait:whether to wait a bit before returning for exposure to register"""
		ret =  self._ssh_command(f"v4l2-ctl -cexposure_absolute={value}")
		if wait:
			time.sleep(0.5)
		return ret

	def set_gain(self, value, wait=True):
		"""Set the gain using -cgain=value
		:param wait:whether to wait before returning for gain to register"""
		ret =  self._ssh_command(f"v4l2-ctl -cgain={value}")
		if wait:
			time.sleep(0.5)
		return ret

	def send_command(self, flags):
		"""Send get/set command to `v4l2-ctl` in the host
		(pass flags and command, e.g. `-cgain=255` or `-Cgain`)"""
		return self._ssh_command(f"v4l2-ctl {flags}").split("Flags:")[-1]

	def get_commands(self):
		"""Return list of available flags from v4l2-ctl"""
		return self._ssh_command("v4l2-ctl --all").split("Flags:")[-1]

class GstSnapshotter():
	"""Uses an opencv camera to take snapshots, using the gstreamer backend. Can configure camera parameters via v4l2
	"""
	def __init__(self, gst_params="v4l2src ! queue! videoconvert ! video/x-raw,framerate=90/1, width=640,height=480 ! queue ! appsink",# max-buffers=10 drop=true",
				 *args, **kwargs):
		"""Init a `cv2.VideoCapture` via gstreamer, disable all automatic parameters
		:param gst_params: params to pass to gstreamer for video stream initialization
		"""

		self.cap = cv2.VideoCapture(gst_params, apiPreference=cv2.CAP_GSTREAMER)

		self._v4l_command(" -c".join([#"-cwhite_balance_temperature_auto=false",
		                   			  "auto_exposure=0",
									  #"exposure_auto_priority=0",
									  ]))
	
	def _v4l_command(self, params, retry=1):
		"""Call v4l2-ctl with the given params, if an error happens retry `retry` times
		:param params:args to pass to v4l2-ctl (try `v4l2-ctl --help`)
		:param retry: times to retry operation"""

		command = ["v4l2-ctl"]
		command.extend(params.split())
		print(f"running {' '.join(command)}")

		for i in range(retry+1):
			try:
				return subprocess.check_output(command).decode()
			except subprocess.CalledProcessError: # If a problem happens, just try again once
				time.sleep(0.2)
	
	def __enter__(self):
		pass

	def __exit__(self, *args, **kwargs):
		pass

	def read(self, retries=1):
		"""Try to read from snapshotter
		:param retries: times to retry"""
		for i in range(retries+1):
			try:
				return self.cap.read()
			except:
				time.sleep(1)

	def set_exposure(self, value, wait=0.5):
		"""Set the exposure using -cexposure=value
		:param wait:the time to wait before returning for exposure to register"""
		self._v4l_command(f"-cexposure_time_absolute={value}")
		if wait:
			time.sleep(wait)

	def set_gain(self, value, wait=True):
		"""Set the gain using -cgain=value
		:param wait:whether to wait before returning for gain to register"""
		self._v4l_command(f"-ciso_sensitivity={value}")
		if wait:
			time.sleep(0.5)


if __name__=="__main__":
	snap = MjpgSnapshotter("http://127.0.0.1:8080/")
	ret, im = snap.read()
	if ret:
		print(im.shape)
		cv2.imwrite("test.jpg", snap.get_img())