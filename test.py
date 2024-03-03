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
img_cannyedge = fe.CannyEdge(img_cannyedge,0.3,0.8)
# img_cv_cd = cv2.Canny(img,0.3*255,0.8*255)
# cv2.imshow("Canny Edge",img_cannyedge)
# cv2.imshow("Canny Edge opencv", img_cv_cd)
# cv2.waitKey(0)

# Harris points detection: Return R for each point
img_harrisPts = np.copy(img)
harrisR = fe.HarrisPts(img_harrisPts, kernel_gas=3, kernel_sobel=3, k=0.04)
img_harrisPts[harrisR>0.01*harrisR.max()] = 255
cv2.imshow("Harris", img_harrisPts)

# im_cv2harris = np.copy(img)
# im_cv2harrisPts = cv2.cornerHarris(im_cv2harris,3,3,0.04)
# im_cv2harris[im_cv2harrisPts>0.01*im_cv2harrisPts.max()] = 255
# cv2.imshow("Harris CV2", im_cv2harris)

# Fast points detection
img_fastPts = np.copy(img)
img_fastPts = fe.FastPts(img_fastPts, consective_pts=9, th=50)
cv2.imshow("Fast", img_fastPts)

# im_cv2fast = np.copy(img)
# im_cv2fastpts = cv2.FastFeatureDetector().create(threshold=50).detect(im_cv2fast)
# im_cv2fast = cv2.drawKeypoints(im_cv2fast, im_cv2fastpts,None,(0,255,0))
# cv2.imshow("FAST cv2", im_cv2fast)

# SIFT detection
# img_sift = np.copy(img)
# fe.SIFT(img)

# HoughLines
img = cv2.imread("data/test.jpg", cv2.IMREAD_GRAYSCALE)
img_HoughLines = np.copy(img)
img_HoughLines, lines = fe.HoughLine(img_HoughLines)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for l in lines:
    img_hl = cv2.line(img_color, l[0], l[1],[0,255,0],1)
cv2.imshow("HoughLines", img_hl)



cv2.waitKey(0)



