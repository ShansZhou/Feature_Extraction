import numpy as np
import cv2



# TODO: draw descriptors on image, recover markers on source image.

# generate DoG pyramid
# return: dictionary of octaves containing layers of DoG
def GaussianPyramid(img, octaves=9, layers=6, sig=0.5, k=1):

    
    octave_set = {} # save in dictionary
    img_32f = np.float64(img)
    for oct in range(octaves):
        downsamp_scalar = 1.0/np.power(2,oct)
        img_scaled = cv2.resize(img_32f,dsize=None,fx=downsamp_scalar,fy=downsamp_scalar)
        DoGLayer_set = {} # save in dictionary

        for lay in range(1,layers):
            sigOflayer1 = np.power(k-1,lay)*sig
            sigOflayer2 = np.power(k,lay)*sig

            gauss_layer1 = cv2.GaussianBlur(img_scaled, ksize=[3,3], sigmaX=sigOflayer1, sigmaY=sigOflayer1)
            gauss_layer2 = cv2.GaussianBlur(img_scaled, ksize=[3,3], sigmaX=sigOflayer2, sigmaY=sigOflayer2)

            DoGLayer_set[lay] = gauss_layer2 - gauss_layer1

        octave_set[oct] = DoGLayer_set
    
    return octave_set

# locate extrema in each DoG
# input: a dict of octaves
def LocateExtrema(octave_set, src_img, sig=0.5, k=1):
    extrema_mat = np.zeros(4,np.uint16)
    src_img = np.float32(src_img)
    for ot_idx, layers in octave_set.items():
    
        for l_idx,curr_DoG in layers.items():

            # first and last layer are ommited
            if l_idx==0 or l_idx==(len(layers)-1):

                prev_DoG = layers[l_idx-1]
                next_DoG = layers[l_idx+1]

                cols, rows = np.shape(curr_DoG)

                # check neigbour layers of 3x3 pixel, in total comparing to 26 pixels
                for col in range(1, cols-1):
                    for row in range(1, rows-1):
                        isExtremum = True
                        target_pixel = curr_DoG[col,row]

                        for [x,y] in [[-1,-1],[-1,1],[1,1],[1,-1],[1,0],[-1,0],[0,1],[0,-1],[0,0]]:
                            next_pixel = next_DoG[col+x,row+y]
                            prev_pixel = prev_DoG[col+x,row+y]
                            curr_pixel = curr_DoG[col+x,row+y]

                            if target_pixel>= next_pixel and target_pixel >=prev_pixel and target_pixel>=curr_pixel:
                                isExtremum = True
                            else:
                                isExtremum = False
                                break
                        
                        if isExtremum == True:
                            

                            # reject unstable key points
                            downsamp_scalar = 1.0/np.power(2,ot_idx)
                            sigOflayer = np.power(k,l_idx)*sig
                            img_ds = cv2.resize(src_img,dsize=None,fx=downsamp_scalar,fy=downsamp_scalar)
                            gauss_layer = cv2.GaussianBlur(img_ds, ksize=[3,3], sigmaX=sigOflayer, sigmaY=sigOflayer)

                            
                            # apply Sobel operator
                            sobel_kernel_x = np.array([[-1,0,+1],
                                                    [-2,0,+2],
                                                    [-1,0,+1]])
                            
                            sobel_kernel_y = np.array([[+1,+2,+1],
                                                    [ 0, 0, 0],
                                                    [-1,-2,-1]])
                            
                            Gx = cv2.filter2D(gauss_layer, -1, sobel_kernel_x)
                            Gy = cv2.filter2D(gauss_layer, -1, sobel_kernel_y)

                            Gxx = Gx**2 # A
                            Gyy = Gy**2 # B
                            Gxy = Gx*Gy # C

                            ### reject low contrast ###
                            # 2nd order derivative
                            GGx = cv2.filter2D(Gx, -1, sobel_kernel_x)
                            # Tyler series expand
                            contrast_tyler = Gx[col, row] + (1/2)*(GGx[col, row])

                            ### reject edges ###
                            trace_M = Gxx+Gyy
                            det_M = Gxx*Gyy - Gxy**2
                            r = 10.0
                            if (trace_M[col,row]**2 / det_M[col,row]) > ((r+1)**2/r):
                                continue
                            elif contrast_tyler < 0.03 :
                                continue
                            else:
                                extrema_mat = np.row_stack((extrema_mat, np.array([col, row, l_idx, ot_idx])))



    return extrema_mat[1:-1]    

# assign direction to feature points
def AssignDirection(extrema_mat, src_img, sig=0.5, k=1, bins=10):
    new_kps = []
    src_img = np.float32(src_img)
    for kp in extrema_mat:
        cx,cy,l_idx,ot_idx = kp[0],kp[1],kp[2],kp[3]
        downsamp_scalar = 1.0/np.power(2,ot_idx)
        sigOflayer = np.power(k,l_idx)*sig
        img_ds = cv2.resize(src_img,dsize=None,fx=downsamp_scalar,fy=downsamp_scalar)
        gauss_layer = cv2.GaussianBlur(img_ds, ksize=[3,3], sigmaX=sigOflayer, sigmaY=sigOflayer)
        # apply Sobel operator
        sobel_kernel_x = np.array([[-1,0,+1],
                                [-2,0,+2],
                                [-1,0,+1]])
        
        sobel_kernel_y = np.array([[+1,+2,+1],
                                [ 0, 0, 0],
                                [-1,-2,-1]])
        
        Gx = cv2.filter2D(gauss_layer, -1, sobel_kernel_x)
        Gy = cv2.filter2D(gauss_layer, -1, sobel_kernel_y)
        mag = np.sqrt(Gx**2+Gy**2)
        theta = np.arctan2(Gy,Gx)/np.pi*180.0 # convert to degree


        r = np.int8(np.floor(3*1.5*sigOflayer))
        binNum = 360//bins
        hist_directs = np.zeros(binNum, np.float32)
        # generate gaussian kernel
        kernelSize = r*2
        kernel = np.zeros((kernelSize,kernelSize))
        
        for x in range(kernelSize):
            for y in range(kernelSize):
                kx = x-1
                ky = y-1
                kernel[x,y] = (1/2*1.5*sigOflayer**2*np.pi) * np.exp(-((kx**2 + ky**2)/2*1.5*sigOflayer*2))
        kernel = kernel / np.sum(kernel)

        cols, rows = np.shape(img_ds)

        # check pixels in the Circle area, weight each direction of bins
        for wx in range(-r, r):
            for wy in range(-r, r):
                img_x = np.int16(cx + wx)
                img_y = np.int16(cy + wy)

                if img_x < 0 or img_x > cols-1: continue
                elif img_y < 0 or img_y > rows-1:continue

                m, direct = mag[img_x, img_y], theta[img_x, img_y]
                weight = kernel[wx+r, wy+r]* m
                deg_mod = np.mod(direct+360,360)
                curr_binno = np.uint8(deg_mod//bins)
                hist_directs[curr_binno] += weight
        
        max_binno = np.argmax(hist_directs)
        # TODO: intepolating accurate degree from histogram
        main_direct = hist_directs[max_binno] 
        new_kps.append([cx,cy,l_idx,ot_idx, max_binno*(360//bins)])

        for bin, val in enumerate(hist_directs):
            if bin==max_binno: continue

            if val > main_direct*0.8:
                new_kps.append([cx,cy,l_idx,ot_idx, bin*(360//bins)]) # subsidiary directions are too much ???

    
    return new_kps

# create descriptors
def GenDescriptors(feat_pts, src_img, sig=0.5, k=1):
    src_img = np.float32(src_img)
    descriptors =[]
    for kp in feat_pts:
        cx,cy,l_idx,ot_idx, deg = kp[0],kp[1],kp[2],kp[3],kp[4]
        
        # Find Gaussian layer WRT this keypoint
        downsamp_scalar = 1.0/np.power(2,ot_idx)
        sigOflayer = np.power(k,l_idx)*sig
        img_ds = cv2.resize(src_img,dsize=None,fx=downsamp_scalar,fy=downsamp_scalar)
        gauss_layer = cv2.GaussianBlur(img_ds, ksize=[3,3], sigmaX=sigOflayer, sigmaY=sigOflayer)
        # apply Sobel operator
        sobel_kernel_x = np.array([[-1,0,+1],
                                [-2,0,+2],
                                [-1,0,+1]])
        
        sobel_kernel_y = np.array([[+1,+2,+1],
                                [ 0, 0, 0],
                                [-1,-2,-1]])
        
        Gx = cv2.filter2D(gauss_layer, -1, sobel_kernel_x)
        Gy = cv2.filter2D(gauss_layer, -1, sobel_kernel_y)
        mag = np.sqrt(Gx**2+Gy**2)
        theta = np.arctan2(Gy,Gx)*np.pi/180.0 # convert to degree


        # initialize a region of feature with centering at the keypoint
        # rotate all the points in the region to the direction of keypoint
        d = 4
        radius = np.int8(np.round(k*sig*(d+1)*np.sqrt(2)*0.5))

        # cal rotation matrix based on the direction of current keypoint
        cos_theta = np.cos(deg)
        sin_theta = np.sin(deg) 

        # 8 bins for 360 degree
        binsNum = 8
        hist = np.zeros(binsNum,np.float32)

        for rx in range(-radius, radius):
            for ry in range(-radius, radius):

                rx_rot = rx*cos_theta - ry*sin_theta
                ry_rot = rx*sin_theta + ry*cos_theta

                xbin = rx_rot +d/2 -0.5
                ybin = ry_rot +d/2 -0.5

                if xbin>-1.0 and xbin<d and ybin>-1.0 and ybin<d:
                    img_x = cx + rx
                    img_y = cy + ry
                    if img_x>0 and img_x<np.shape(src_img)[0] and img_y>0 and img_y<np.shape(src_img)[1]:
                        theta_rot = (theta[img_x, img_y] - deg) % 360
                        weight_rot = np.exp(-(rx_rot**2 + ry_rot**2)/(d**2*0.5))
                        mag_rot = mag[img_x,img_y]*weight_rot
                        binno = np.uint8(np.mod(theta_rot+360, 360) //45)
                        hist[binno] += mag_rot

        descript = np.concatenate((np.array(kp), hist))
        descriptors.append(descript)
        # print(hist)

    return descriptors

# show feature points on image
def showFeatPts(descriptors, src_img,sig=0.5, k=1):
    
    for descript in descriptors:

        x, y, l_idx, ot_idx, deg = descript[0:5]

        downsamp_scalar = 1.0/np.power(2,ot_idx)
        scale = sig*k*(np.power(2,ot_idx))*0.8*(np.power(2,l_idx))*0.2
        draw_radius =np.int16(scale)

        # draw location
        cv2.circle(src_img, np.int16((x/downsamp_scalar,y/downsamp_scalar)), draw_radius, (0,255,0), thickness=1)
        
        # draw direction
        rad = deg/180*np.pi
        dest_pts = [x/downsamp_scalar+np.sin(rad)*draw_radius, y/downsamp_scalar+np.cos(rad)*draw_radius]
        cv2.line(src_img, np.int16((x/downsamp_scalar,y/downsamp_scalar)),np.int16((dest_pts[0], dest_pts[1])),(255,0,0),1)
    
    
    cv2.namedWindow("SIFT", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("SIFT", src_img)
    cv2.waitKey(0)





##############################################################
#### Test part ####
    
# img = cv2.imread("./data/leno.jpg", cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, dsize = None, fx=0.3, fy=0.3)

# # generate DoG pyramid
# src_img = np.copy(img)
# octave_set = GaussianPyramid(src_img, octaves=9, layers=6, sig=0.5, k=1)
# # locating feature points (extrema)
# extrema_mat = LocateExtrema(octave_set, src_img, sig=0.5, k=1)
# # assgin direction to feature points
# feat_pts = AssignDirection(extrema_mat, src_img, sig=0.5, k=1, bins=10)
# # generate descrptors
# descriptors = GenDescriptors(feat_pts, src_img, sig=0.5, k=1)
# # show feature points
# showFeatPts(descriptors, src_img, sig=0.5, k=1)