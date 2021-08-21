from snapshot import MjpgSnapshotterExp as Snapshotter
import cv2
import pickle
import time
import numpy as np
from MotionController import MotionController

def gen_points_grid(z, xlims=(70,180), ylims=(50,140), shape=(4,4), optimize_route=True):
    """Generate points in a grid
    @param shape of the line in (rows, cols)
    @param xlims first and last points of the grid in x
    @param ylims first and last points of the grid in y
    @param num number of points in each line
    """
    x = np.linspace(*xlims, num=shape[1])
    y = np.linspace(*ylims, num=shape[0])
    xp, yp = np.round(np.meshgrid(x, y),2)
    
    if optimize_route:
        xp[1::2]=xp[1::2][:,::-1]

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
    cam = Snapshotter("http://localhost/webcam/")

    # Per Point
    exposures=[50, 100, 500, 1000]

    HIGH_Z = 180
    LOW_Z = 100

    points=[]
    for l in (gen_points_grid(z, shape=(3,3)) for z in np.linspace(LOW_Z,HIGH_Z,num=5)):
        points.extend(l)

    points.extend(list((125, 125, z) for z in range(HIGH_Z,LOW_Z, -5)))

    print(f"Generating pics of {len(points)} points: ")
    print(points)
    entries = []
    with c:
        c.send_command("G28")
        c.send_command("G0 F5000")
        for p in points:
            c.move_to(*p)
            pics = []
            print(f"point={p}")
            for e in exposures:
                print(f"\texposure={e}")
                cam.set_exposure(e)
                time.sleep(0.5)
                #ret, pic = cam.read()
                pic = cam.get_raw()
                if not pic.status_code==200:
                    print("FAILED!")
                    break
                pics.append(pic)

            entry = dict(point=p, pictures=pics, exposures=exposures)
            entries.append(entry)

    with open("output.pkl", "wb") as f:
        pickle.dump(entries, f)
