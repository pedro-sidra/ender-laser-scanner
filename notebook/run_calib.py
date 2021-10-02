from calib import calibrate_charuco, load_coefficients, save_coefficients
import cv2

# Parameters
IMAGES_DIR = r'/home/freitas/TCC/ender-laser-scanner/pics/calibref'
IMAGES_FORMAT = 'jpg'
# Dimensions in cm
MARKER_LENGTH = 18
SQUARE_LENGTH = 25

SIZE = (8,8)


# Calibrate 
ret, mtx, dist, rvecs, tvecs = calibrate_charuco(
    IMAGES_DIR, 
    IMAGES_FORMAT,
    MARKER_LENGTH,
    SQUARE_LENGTH,
    plot=False
)
print(ret)
# Save coefficients into a file
save_coefficients(mtx, dist, "calibration_charuco.yml")

# Load coefficients
mtx, dist = load_coefficients('calibration_charuco.yml')

path = r"/home/freitas/TCC/ender-laser-scanner/pics/calibref/WIN_20210901_21_41_26_Pro.jpg"

original = cv2.imread(path)
dst = cv2.undistort(original, mtx, dist, None, mtx)
cv2.imshow("Img", dst)
cv2.waitKey()


# With prior
ret, mtx, dist, rvecs, tvecs = calibrate_charuco(
    IMAGES_DIR, 
    IMAGES_FORMAT,
    MARKER_LENGTH,
    SQUARE_LENGTH,
    prior = (mtx, dist),
    plot=True
)
print(ret)
save_coefficients(mtx, dist, "calibration_charuco.yml")