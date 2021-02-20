import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import cv2

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img,vertices_l, vertices_r):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_cnt = img.shape[2]
        ignore_mask_color = (255,) * channel_cnt # (255,255,255)
    else:
        ignore_mask_color = 255 # 255

    cv2.fillPoly(mask,vertices_l, ignore_mask_color)
    cv2.fillPoly(mask,vertices_r, ignore_mask_color)
    masked_image = cv2.bitwise_and(img,mask) 

    return masked_image
#######################################

def draw_lines(img, lines, color = [255,0,0], thickness = 5):
    if lines is not None:
        for line in lines:
            #print(line)
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1,y1), (x2,y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength = min_line_len, maxLineGap = max_line_gap)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    draw_lines(line_img,lines)
    return line_img

#######################################

def weighted_img(img, initial_img, alpha = 1., beta = 1., lamb = 0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, lamb)



dir_ = '/home/ubuntu/Documents/cm/contest_frame/'

for i in range(112):
    num = str(i)
    name = dir_ + "frame" + num + ".jpg"
    img = mpimg.imread(name)
    
    gray = grayscale(img)

    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)
    low_threshold = 50
    high_threshold = 200
    edges = canny(blur_gray, low_threshold, high_threshold)
        
    imshape = img.shape # (1080,1920,3)
    
    vertices_l = np.array([[(76,800),(538,441),(589,441),(400,800)]], dtype = np.int32) # bottom left -> top left -> top right -> bottom right
    vertices_r = np.array([[(838,800),(640,441),(667,441),(1200,800)]], dtype = np.int32)
    
    mask = region_of_interest(edges, vertices_l, vertices_r)

    rho = 2
    theta = np.pi/180
    threshold = 90
    min_line_len = 120
    max_line_gap = 150

    
    height, width = imshape[:2]

    src = np.float32([[0,0],[1200,0],[0,height],[980,height]])
    #dst = np.float32([[847,617],[1095,617],[282,886],[1602,886]]) # roi
    #dst = np.float32([[860,627],[1074,627],[494,880],[1440,880]])
    dst = np.float32([[564,426],[662,423],[0,800],[1200,800]]) # top left -> top right-> bottom left -> bottom right
    
    M=cv2.getPerspectiveTransform(dst,src)  # transformation
    #Minv = cv2.getPerspectiveTransform(dst,src) # inverse transformation
    warped_img = cv2.warpPerspective(img, M, (width, height))


    if i == 57  :
        plt.figure(figsize = (10,8))
        plt.imshow(warped_img, cmap = 'gray')

plt.show()