"""
generate_pics

Generate (positin, picture) pairs by moving a CNC with the `MotionController`
and taking snapshots with a `Snapshotter`
"""
import snapshot
import json
from os.path import join
import subprocess
from pathlib import Path
import pty, os 
import argparse
from MotionController import MotionController, FakeMotionController
import threading
import cv2
import time
import numpy as np
import sys
import argparse

def gen_points_grid(z, xlims=(70,180), ylims=(50,140), shape=(4,4), optimize_route=True):
    """Generate points in a grid
    :param shape:of the line in (rows, cols)
    :param xlims:first and last points of the grid in x
    :param ylims:first and last points of the grid in y
    :param num:number of points in each line
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

def gen_points_line(x, z, ylims=(90, 125), num=50):
    yp = np.linspace(*ylims, num=num)

    yp = yp.flatten()
    xp = np.ones_like(yp) * x
    zp = np.ones_like(yp) * z

    return list(zip(xp,yp,zp))

def hdr(images, times, gamma=2.2):
    # Merge exposures to HDR image
    merge_debevec = cv2.createMergeDebevec()
    hdr_debevec = merge_debevec.process(images, times=np.array(times, dtype=np.float32))

    tonemap1 = cv2.createTonemap(gamma=gamma)
    res_debevec = tonemap1.process(hdr_debevec.copy())
    res_debevec = np.clip(res_debevec*255, 0, 255).astype('uint8')
    return res_debevec

def get_args():
    """Command-line parser for this script"""
    if sys.version_info<(3,5,0):
        sys.stderr.write("You need python 3.5 or later to run this script\n")
        sys.exit(1)

    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    p.add_argument("output",
                   help="Path to the output")
    p.add_argument("--feed", type=int,
                   help="Feedrate to use for all movements", 
                   default=5000)
    p.add_argument("--nohome", action="store_true",
                   help="Don't send G28 at the start")
                   
    p.add_argument("--once", action="store_true",
                   help="Take one picture and quit")

    p.add_argument("--save_exps", action="store_true",
                   help="save all exposures")

    p.add_argument("--calib", action="store_true",
                   help="Run the calib sequence")
    p.add_argument("--scan", action="store_true",
                   help="Run the scan sequence")
                   
        
    return(p.parse_args())


if __name__=="__main__":
    # Cmdline arguments
    args = get_args()

    #------CONFIG------------------------------------------
    # Take exposures with this absolute config at each point
    exposures=[50, 100, 500, 1000]
    # Top and bottom of Z range
    LOW_Z, HIGH_Z = (100, 180)

    #------POINTS------------------------------------------
    points=[]

    if args.calib:
        for l in (gen_points_grid(z, shape=(3,3)) for z in np.linspace(LOW_Z,HIGH_Z,num=5)):
            points.extend(l)

        points.extend(list((125, 125, z) for z in range(HIGH_Z,LOW_Z, -5)))
    
    if args.scan:
        points.extend(gen_points_line(125, (HIGH_Z + LOW_Z)/2))
    
    if args.once:
        points = [[1,2,3]]

    #------OUTPUT PATH-------------------------------------
    outpath = Path(args.output)
    outpath.mkdir(parents=True, exist_ok=True)

    if args.save_exps:
        outpath.joinpath("exposures").mkdir(parents=True, exist_ok=True)

    print(f"Generating pics of {len(points)} points: ")
    print(points)
    img_coords = {}

    #------IO OBJECTS--------------------------------------
    # To control movement with GCode
    if args.once:
        motion = FakeMotionController()
    else:
        motion = MotionController()
    # To take pictures
    cam = snapshot.GstSnapshotter()
    # Set initial params
    cam.set_exposure(50)
    cam.set_gain(1)

    #----RUN-----------------------------------------------
    with motion, cam:

        if not args.nohome:
            motion.send_command("G28")
        motion.send_command(f"G0 F{args.feed}")

        for i, p in enumerate(points):

            # ------ MOVE TO THE POINT
            motion.move_to(*p)
            print(f"point={p}")

            # ------ TAKE PICTURES FOR EACH EXPOSURE
            pics = []
            ret, pic = cam.read()
            for e in exposures:
                cam.set_exposure(e, wait=1)
                print(f"\texposure={e}")
                ret, pic = cam.read()
                pics.append(pic)
            
            # ------ SAVE EXPOSURES
            if args.save_exps:
                for i_exp, pic in enumerate(pics):
                    cv2.imwrite(join(outpath, "exposures", f"{i}_exposure{i_exp}.png"), pic)

            # ------ MAKE HDR PICTURE AND SAVE
            pic = hdr(pics, exposures)
            filename= f"{i}.png"
            cv2.imwrite(join(outpath, filename), pic)

            img_coords[filename] = p
    
    with open(join(outpath,"points.json"), "w") as f:
        f.write(json.dumps(img_coords))

    print(subprocess.check_output(f"zip -r {outpath}.zip {outpath}".split()))
