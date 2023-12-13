import numpy as np
import cv2



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
    img = cv2.GaussianBlur(img, ksize=(5,5), sigmaX=0.8, sigmaY=0.8)

    # apply Sobel operator
    sobel_kernel_x = np.array([[-1,0,+1],
                               [-2,0,+2],
                               [-1,0,+1]])
    
    sobel_kernel_y = np.array([[+1,+2,+1],
                               [ 0, 0, 0],
                               [-1,-2,-1]])
    
    Gx = cv2.filter2D(img, -1, sobel_kernel_x)
    Gy = cv2.filter2D(img, -1, sobel_kernel_y)

    theta = np.arctan2(Gy, Gx)/np.pi*180.0
    mag = np.sqrt(Gx**2+Gy**2)
    
    # 8 direction of bins
    direct = np.array([[-1,-1],[1,-1],[+1,-1],[+1,0],[+1,+1],[0,+1],[-1,+1],[-1,0]]) 
    
    # iterate all theta in the image
    cols, rows = np.shape(theta)

    for col in range(1,cols-1):
        for row in range(1,rows-1):
            deg = theta[col, row]
            deg_mod = np.mod(np.int16(deg+360), 360)
            degInSeg = deg_mod//45

            # Non maximun suppression
            offsetX1 = direct[degInSeg][0]
            offsetY1 = direct[degInSeg][1]

            offsetX2 = direct[np.mod(degInSeg+4,8)][0]
            offsetY2 = direct[np.mod(degInSeg+4,8)][1]

            deg1 = theta[col+offsetX1, row+offsetY1]
            deg2 = theta[col+offsetX2, row+offsetY2]

            if mag[col,row] >= mag[col+offsetX1,row+offsetY1] and mag[col,row] >= mag[col+offsetX2, row+offsetY2]:
                pass
            else:
                img[col,row] = 0

    # Fuzzy threshold
    mask = np.ones((cols, rows), np.uint8)

    img_32f = np.float32(img)/255.0
    mask[img_32f<lower_th] = 0
    mask[img_32f>higher_th] = 2

    Neigbs8 = direct

    for col in range(1, cols - 1):
        for row in range(1, rows - 1):
            if mask[col, row] ==1:
                isRemained = False
                for [x_off, y_off] in Neigbs8:
                    if mask[col+x_off, row+y_off] ==1:
                        isRemained = True
                        break

                if isRemained:
                    mask[col, row] = 1
                else:
                    mask[col, row] = 0

            if mask[col, row] ==2:
                mask[col, row] = 1

    img = img*mask

    return img
                    
# Corner detection



    


            









