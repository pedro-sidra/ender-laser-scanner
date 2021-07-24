from snapshot import MjpgSnapshotter as Snapshotter
import cv2
import numpy as np
from MotionController import MotionController

def gen_points(z, num=4):
    x = np.linspace(70, 180, num=num)
    y = np.linspace(50, 140, num=num)
    xp, yp = np.round(np.meshgrid(x, y),2)

    xp = xp.flatten()
    yp = yp.flatten()
    zp = np.ones_like(xp) * z

    return list(zip(xp,yp,zp))

if __name__=="__main__":
    c = MotionController()
    cam = Snapshotter("http://ender3.local/webcam/")
    pics=[]
    points=[]
    for l in (gen_points(z, num=3) for z in (250, 200, 150, 100)):
        points.extend(l)

    print(f"Generating pics of {len(points)} points: ")
    print(points)
    with c:
        c.send_command("G28")
        for p in points:
            c.move_to(*p)
            ret, pic = cam.read()
            if not ret:
                break
            pics.append(pic)

    for i, p in enumerate(pics):
        cv2.imwrite(f"pics/img_{i}.jpg", p)