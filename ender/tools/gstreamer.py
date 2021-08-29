import cv2
import time
import subprocess

def command(cmd):
    """Call command, retry once if failure"""

    command = cmd.split()
    try:
        return subprocess.check_output(command).decode()
    except subprocess.CalledProcessError:
        time.sleep(0.5)
        return subprocess.check_output(command).decode()

cameraPipeline ="v4l2src ! video/x-raw,framerate=5/1, width=(int)1280,height=(int)960 ! videoconvert ! appsink max-buffers=1 drop=true"

# cameraPipeline+="video/x-raw, format=YUYV, framerate=5/1, width=(int)1280,height=(int)720 ! "
# cameraPipeline+="appsink"
#  extra-controls=\"c,exposure_auto=1,exposure_absolute=50\" 
cap = cv2.VideoCapture(cameraPipeline, apiPreference=cv2.CAP_GSTREAMER)
ret, img = cap.read()
cv2.imwrite(f"test.png", img)

command("v4l2-ctl -cwhite_balance_temperature_auto=0")
command("v4l2-ctl -cexposure_auto=1")
command("v4l2-ctl -cexposure_auto_priority=0")
command("v4l2-ctl -cwhite_balance_temperature=0")
command("v4l2-ctl -cgain=1")

exposures=[10, 100, 500, 1000]
for e in exposures:
    command(f"v4l2-ctl -cexposure_absolute={e}")
    # for i in range(10):
    time.sleep(0.8)
    ret, img = cap.read()
    print(img.shape)
    cv2.imwrite(f"{e}.png", img)
