import requests
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


if __name__=="__main__":
	snap = MjpgSnapshotter("http://127.0.0.1:8080/")
	ret, im = snap.read()
	if ret:
		print(im.shape)
		cv2.imwrite("test.jpg", snap.get_img())