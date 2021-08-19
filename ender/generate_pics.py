from snapshot import MjpgSnapshotterExp as Snapshotter
import cv2
import pickle
import numpy as np
from MotionController import MotionController

def gen_points_grid(z, num=4):
    x = np.linspace(70, 180, num=num)
    y = np.linspace(50, 140, num=num)
    xp, yp = np.round(np.meshgrid(x, y),2)

    xp = xp.flatten()
    yp = yp.flatten()
    zp = np.ones_like(xp) * z

    return list(zip(xp,yp,zp))

def gen_points_line(x, z, num=50):
    yp = np.linspace(10, 200, num=num)

    yp = yp.flatten()
    xp = np.ones_like(yp) * x
    zp = np.ones_like(yp) * z

    return list(zip(xp,yp,zp))


"""
[
    {
        "location": coords,
        "pictures":[pics],
        "exposures":[exposures]
    },
    ...
    {
        "location": coords,
        "pictures":[pics],
        "exposures":[exposures]
    }
]
"""
if __name__=="__main__":
    c = MotionController()
    cam = Snapshotter("http://ender3.local/webcam/")

    # Per Point
    exposures=[25, 50, 100, 500, 1000, 10000]

    points = gen_points_line(x=130, z=230)
    # for l in (gen_points_grid(z, num=3) for z in (250, 200, 150, 100)):
    #     points.extend(l)

    print(f"Generating pics of {len(points)} points: ")
    print(points)
    entries = []
    with c:
        c.send_command("G28")
        for p in points:
            c.move_to(*p)
            pics = []
            for e in exposures:
                cam.set_exposure(e)
                ret, pic = cam.read()
                if not ret:
                    break
                pics.append(pic)

            entry = dict(point=p, pictures=pics, exposures=exposures)
            entries.append(entry)

    with open("output.pkl", "wb") as f:
        pickle.dump(entries, f)