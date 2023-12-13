import feature_extraction as fe
import cv2
import numpy as np


# load gray img
img = cv2.imread("./data/leno.jpg", cv2.IMREAD_GRAYSCALE)
# dsize = np.uint16([np.shape(img)[1]*0.5,np.shape(img)[0]*0.5])
img = cv2.resize(img, dsize = None, fx=0.5, fy=0.5)

# cal gradient
gx, gy, mag, theta = fe.cal_gradient(img)

theta_deg = theta/np.pi*180.0

cv2.imshow("Gx", gx)
cv2.imshow("Gy", gy)

# Histogram of Oriented Gradient
blocks = fe.HOG(theta_deg)
print("block: ",np.shape(blocks)[0])

# Canny edge dection
img_cannyedge = fe.CannyEdge(img)
cv2.imshow("Canny Edge", img_cannyedge)


cv2.waitKey(0)



