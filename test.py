import feature_extraction as fe
import cv2
import numpy as np


# load gray img
img = cv2.imread("./data/leno.jpg", cv2.IMREAD_GRAYSCALE)
# dsize = np.uint16([np.shape(img)[1]*0.5,np.shape(img)[0]*0.5])
img = cv2.resize(img, dsize = None, fx=0.5, fy=0.5)
cv2.imshow("src", img)
# cal gradient
gx, gy, mag, theta = fe.cal_gradient(img)

theta_deg = theta/np.pi*180.0

cv2.imshow("Gx", gx)
cv2.imshow("Gy", gy)

# Histogram of Oriented Gradient
blocks = fe.HOG(theta_deg)
print("block: ",np.shape(blocks)[0])

#####################

# Canny edge dection
img_cannyedge = np.copy(img)
img_cannyedge = fe.CannyEdge(img_cannyedge)
cv2.imshow("Canny Edge", img_cannyedge)

# Harris points detection
img_harrisPts = np.copy(img)
img_harrisPts = fe.HarrisPts(img_harrisPts, windowsize=[3,3], k=0.06)
cv2.imshow("Harris", img_harrisPts)

# Fast points detection
img_fastPts = np.copy(img)
img_fastPts = fe.FastPts(img_fastPts, consective_pts=9, th=20)
cv2.imshow("Fast", img_fastPts)



cv2.waitKey(0)



