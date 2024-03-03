import numpy as np
import cv2
import SIFT


# calculate gradient
def cal_gradient(img):
    tmp = np.zeros((np.shape(img)[0],1))
    tmp = np.column_stack((tmp, img))
    offset_img = np.delete(tmp,-1, axis=1)

    gx = offset_img - img

    tmp = np.zeros(np.shape(img)[1])
    tmp = np.row_stack((tmp,img))
    offset_img_col = tmp[:-1,:]

    gy = offset_img_col - img

    mag = np.sqrt(gx**2+ gy**2)

    theta = np.arctan2(gy, gx)

    return gx, gy, mag, theta

# Histogram of Oriented Gradient
# input: the graident degree of image
def HOG(grad_deg, block=[2,2], stride = 12, bins=9):
    
    # save gradient distribution in every block: 9 bins for each block
    blocks = np.zeros(bins, np.uint16)

    # each block contains 2x2 cells; each cells contains 6x6 pixels
    cells = [6,6]

    cols, rows = np.shape(grad_deg)
    
    # iterate each blocks
    for col in range(0, cols - block[0]*cells[0], stride):
        for row in range(0, rows - block[1]*cells[1], stride):
            segs = np.zeros(bins, np.uint16)
            for x in range(cells[0]):
                for y in range(cells[1]):
                    c_x = col+x
                    r_y = row+y
                    deg = np.int16(grad_deg[c_x,r_y])
                    deg_mod = np.mod(deg+360,360)
                    degInSeg = deg_mod//40

                    if degInSeg == 9:
                        degInSeg = 8

                    segs[degInSeg] = segs[degInSeg]+1
            blocks = np.row_stack((blocks,segs))

    # remove first row
    return blocks[1:-1]
    
# Canny Edge detection
def CannyEdge(img,lower_th=0.3, higher_th=0.8):

    # apply Gaussian fliter
    img_gas = cv2.GaussianBlur(img/255.0, ksize=(3,3), sigmaX=0.5, sigmaY=0.5)

    # apply Sobel operator
    Gx = cv2.Sobel(img_gas, -1, 1,0)
    Gy = cv2.Sobel(img_gas, -1, 0,1)

    theta = np.arctan2(Gy, Gx)
    mag = np.sqrt(Gx**2+Gy**2)
    
    # 8 direction of bins
    direct = np.array([[-1,-1],[0,-1],[+1,-1],[+1,0],[+1,+1],[0,+1],[-1,+1],[-1,0]]) 
    # iterate all theta in the image
    rows, cols= np.shape(img_gas)
    # Non maximun suppression - [Grident Mag]
    for row in range(1,rows-1):
        for col in range(1,cols-1):
            deg = theta[row,col]*180/np.pi
            
            offX1, offY1, offX2, offY2 = 0,0,0,0
            if  (-22.5<deg<=22.5) or (-180<=deg<=-157.5) or (157.5<deg<=180):
                offX1, offY1, offX2, offY2 = 0,1,0,-1
            elif (22.5<deg<=67.5) or (-157.5<deg<=-112.5):
                offX1, offY1, offX2, offY2 = 1,1,-1,-1
            elif (67.5<deg<=112.5) or (-112.5<deg<=-67.5):
                offX1, offY1, offX2, offY2 = 1,0,-1,0
            elif (112.5<deg<=157.5) or (-67.5<deg<=-22.5):
                offX1, offY1, offX2, offY2 = +1,-1,-1,+1
  
            if mag[row,col] >= mag[row+offY1,col+offX1] and mag[row,col] >= mag[row+offY2,col+offX2]:
                pass
            else:
                mag[row,col]=0.0

    mask = np.ones(img_gas.shape,np.uint8)
    # Fuzzy threshold
    Neigbs8 = direct
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            
            if mag[row, col] < lower_th:
                mask[row, col] = 0
            elif mag[row, col] > higher_th:
                mask[row, col] = 255
            else:
                for [x_off, y_off] in Neigbs8:
                    
                    if row+y_off<0 or row+y_off>=rows or col+x_off<0 or col+x_off>=cols:continue
                    
                    if mag[row+y_off, col+x_off] > higher_th:
                        mask[row, col] = 255
                        break
                    else:
                        mask[row, col] = 0

    return mask
                    
# Harris corner
def HarrisPts(img, kernel_gas=3, kernel_sobel=3, k=0.06):
    
    img_norm = np.float32(img)/255.0
    
    # apply Sobel operator
    Gx = cv2.Sobel(img_norm, -1, 1, 0, kernel_sobel)
    Gy = cv2.Sobel(img_norm, -1, 0, 1, kernel_sobel)

    Gxx = Gx**2 # A
    Gyy = Gy**2 # B
    Gxy = Gx*Gy # C

    sig = 1.3 # correspond to cv2
    A = cv2.GaussianBlur(Gxx,(kernel_gas,kernel_gas),sig)
    B = cv2.GaussianBlur(Gyy,(kernel_gas,kernel_gas),sig)
    C = cv2.GaussianBlur(Gxy,(kernel_gas,kernel_gas),sig)

    det_M = A*B - (C**2)
    trace_M = A+B
    R = det_M - k*(trace_M**2)

    return R
    
# Fast corner
def FastPts(img, consective_pts = 9, th = 50):

    circle_kernel = np.array([
              [0,-4],[+1,-4],[+3,-3],[+4,-2],
              [+4,0],[+4,+1],[+3,+3],[+2,+4],
              [0,+4],[-1,+4],[-3,+3],[-4,+2],
              [-4,0],[-4,-1],[-3,-3],[-2,-4],
              ])
    
    rows,cols = np.shape(img)
    mask = np.zeros(np.shape(img))
    for row in range(4, rows-4):
        for col in range(4, cols-4):
            acc = 0
            Ip = img[row, col]
            LT = Ip - th
            HT = Ip + th

            # check 16 pts connectivities around the pixel
            for [offX,offY] in circle_kernel:
                Ip2x = img[row+offY, col+offX]
                
                if Ip2x > LT and Ip2x < HT: # the pixel is similar to center, then it is not corner
                    break
                else:
                    acc = acc+1
                    if acc >= consective_pts: # surfficient consective points 
                        break
            if acc >= consective_pts:
                img[row, col] = 255
                mask[row, col] = 1
    
    print("Fast points:", np.sum(mask))

    return img

# SIFT points
def runSIFT(img):
    # generate DoG pyramid
    src_img = np.copy(img)
    octave_set = SIFT.GaussianPyramid(src_img, octaves=9, layers=6, sig=0.5, k=1)
    # locating feature points (extrema)
    extrema_mat = SIFT.LocateExtrema(octave_set, src_img, sig=0.5, k=1)
    # assgin direction to feature points
    feat_pts = SIFT.AssignDirection(extrema_mat, src_img, sig=0.5, k=1, bins=10)
    # generate descrptors
    descriptors = SIFT.GenDescriptors(feat_pts, src_img, sig=0.5, k=1)
    # show feature points
    SIFT.showFeatPts(descriptors, src_img, sig=0.5, k=1)


# Hough Transform
def HoughLine(img, k=100):

    src_img = np.copy(img)

    img_edge = cv2.Canny(src_img, 250, 253)
    # cv2.imshow("testcanny", img_edge)
    # cv2.waitKey(0)

    rows, cols = np.shape(img_edge)
    theta = 180
    maxLine = np.uint16(np.sqrt(rows**2+cols**2))
    # hough_mat = np.zeros((theta, maxLine))
    hough_dict = {} # store lines params
    for row in range(rows):
        for col in range(cols):
            if img_edge[row,col] ==0: continue

            for t in range(theta):
                rad = t/180*np.pi
                p = np.int16(col*np.cos(rad) + row*np.sin(rad)) # p probablely could be negative
                
                if hough_dict.get((t,p)) == None:
                    hough_dict[(t,p)] =1
                else:
                    hough_dict[(t,p)] +=1

    # iterate each line(val>k), find locations in image
    lines = []
    for (t, p), val in hough_dict.items():
        if val < k : continue
        
        # find all possible pixel locations along the line
        maxX = 0
        minX = maxLine
        pt_max = [0,0]
        pt_min = [0,0]
        for col in range(cols):
            rad = t/180*np.pi

            if np.sin(rad) == 0: continue

            # known col, compute row wrt P equation
            row = np.uint16((p - col*np.cos(rad)) / np.sin(rad))

            if 0<=row<rows and img_edge[row,col]!=0:
                src_img[row, col] = 255
                
                if maxX < row:
                    maxX = row
                    pt_max = [col,row]
                if minX > row:
                    minX = row    
                    pt_min = [col,row]
        
        lines.append([pt_max,pt_min])


    return src_img, lines
                

