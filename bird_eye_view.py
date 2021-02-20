import cv2
import time
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import math

class lane_detect():

    def grayscale(self,img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def gaussian_blur(self,img, kernel_size):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def canny(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)

    def region_of_interest(self, img,vertices):
        mask = np.zeros_like(img)

        if len(img.shape) > 2:
            channel_cnt = img.shape[2]
            ignore_mask_color = (255,) * channel_cnt # (255,255,255)
        else:
            ignore_mask_color = 255 # 255

        cv2.fillPoly(mask, vertices, ignore_mask_color)
        #cv2.fillPoly(mask, vertices_1, ignore_mask_color)


        masked_image = cv2.bitwise_and(img,mask)

        return masked_image


    def draw_lines(self, img, lines, fline_lst_r, fline_lst_l, line_lst_l, line_lst_r, color = [255,0,0], thickness = 5,):


        if lines is not None:

            for line in lines:

                for x1,y1,x2,y2 in line:
                    #print(line)
                    cv2.line(img, (x1,y1), (x2,y2), color, thickness)
                [x1,y1,x2,y2] = line[0][:]


                if x1 > 600:

                    #small_lst = [x1,y1]
                    line_lst_r.append([x1,y1])
                    if len(line_lst_r) == 4:
                        fline_lst_r = line_lst_r

                        line_lst_r = []
                        return fline_lst_r, fline_lst_l, line_lst_r, line_lst_l
                    else:
                        pass


                        return fline_lst_r, fline_lst_l, line_lst_r, line_lst_l

                else:

                    #small_lst = [x2,y2]
                    line_lst_l.append([x2,y2])
                    if len(line_lst_l) == 4:

                        fline_lst_l = line_lst_l

                        line_lst_l = []


                        return fline_lst_r, fline_lst_l, line_lst_r, line_lst_l
                    else:
                        pass


                        return fline_lst_r, fline_lst_l, line_lst_r, line_lst_l
        else:
             # line is Nonetype
            pass

            return fline_lst_r, fline_lst_l, line_lst_r, line_lst_l # final_lst can be (line_lst_1 or line_lst_r or fline_lst_l or fline_lst_r)



    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap, fline_lst_r, fline_lst_l, line_lst_l, line_lst_r):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength = min_line_len, maxLineGap = max_line_gap)
        print(lines)

        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
        fline_lst_r, fline_lst_l, line_lst_r, line_lst_l = self.draw_lines(img, lines, fline_lst_r, fline_lst_l, line_lst_l, line_lst_r)
        return line_img, fline_lst_r, fline_lst_l, line_lst_r, line_lst_l


    def weighted_img(self, img, initial_img, alpha = 1., beta = 1., lamb = 0.):
        return cv2.addWeighted(initial_img, alpha, img, beta, lamb)

    def abcd_func(self, fline_lst):

        M = np.ones((4,4))
        y = np.zeros(4)
        y_t = y.reshape((4,1))
        s_list = 0.00001*np.random.rand(4,4)
        for i in range(len(M)):
            y[i] = (fline_lst[i][1]) # y_t
            for j in range(len(M)):
                if j == 0:
                    pass
                else:
                    flst = int(fline_lst[i][0])
                    M[i][j] = math.pow(flst, j)
        M_inverse = np.linalg.inv(M + s_list) # 4x4 matrix
        abcd = np.dot(M_inverse,y_t)
        return abcd


    def callback(self):
    

        vid = "/home/ubuntu/Documents/cm/movie.mp4"
        cap = cv2.VideoCapture(vid)
        ##

        kernel_size = 5
        low_threshold = 50
        high_threshold = 200


        vertices_l = np.array([[(76,800),(538,441),(589,441),(400,800)]], dtype = np.int32) # bottom left -> top left -> top right -> bottom right
        vertices_r = np.array([[(838,800),(640,441),(667,441),(1200,800)]], dtype = np.int32)


        rho = 2
        theta = np.pi/180
        threshold = 100
        min_line_len = 120
        max_line_gap = 150

        fline_lst_l = []
        fline_lst_r = []
        line_lst_l = []
        line_lst_r = []


        while True:
            if(cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                cap.open("vid")

            ret,frame = cap.read()
            #img = frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert frame to RGB img
            print(type(img))



            imshape = img.shape # (1080,1920,3)

            height, width = imshape[:2]

            src = np.float32([[0,0],[1200,0],[0,height],[980,height]])
            #dst = np.float32([[807,630],[1090,631],[376,886],[1521,886]]) # roi
            #dst = np.float32([[847,617],[1095,617],[282,886],[1602,886]]) # roi
            dst = np.float32([[564,426],[662,423],[0,800],[1200,800]]) # top left -> top right-> bottom left -> bottom right


            M = cv2.getPerspectiveTransform(dst,src)  # transformation
            #Minv = cv2.getPerspectiveTransform(dst,src) # inverse transformation
            warped_img = cv2.warpPerspective(img, M, (width, height))  # Bird-eye view img

            ##########################################################

            gray = self.grayscale(warped_img)


            blur_gray = self.gaussian_blur(gray, kernel_size)


            edges = self.canny(blur_gray, low_threshold, high_threshold)


            mask_l = self.region_of_interest(edges, vertices_l)
            mask_r = self.region_of_interest(edges, vertices_r)


            lines_l, fline_lst_r, fline_lst_l, line_lst_r, line_lst_l = self.hough_lines(mask_l, rho, theta, threshold, min_line_len, max_line_gap,fline_lst_r, fline_lst_l,
            line_lst_l, line_lst_r)



            lines_r, fline_lst_r, fline_lst_l, line_lst_r, line_lst_l= self.hough_lines(mask_r, rho, theta, threshold, min_line_len, max_line_gap,fline_lst_r, fline_lst_l,
            line_lst_l, line_lst_r)






        #


            if len(fline_lst_r) == 4:
                abcd = self.abcd_func(fline_lst_r)
                print(abcd)


            if len(fline_lst_l) == 4:
                abcd = self.abcd_func(fline_lst_l)
                print(abcd)


            lines_final = self.weighted_img(lines_l, warped_img, alpha = 1., beta = 1., lamb = 0.)  # color lines + Bird-eye view
            lines_final = self.weighted_img(lines_r, lines_final, alpha = 1., beta = 1., lamb = 0.)  # color lines + Bird-eye view
            final = cv2.resize(lines_final,dsize=(640,480),interpolation=cv2.INTER_AREA)  # resize output vid
        #################################################################

            cv2.imshow("result",img)
            #plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) # Show results
            #plt.show()

            if cv2.waitKey(20) > 0:
                break

        cap.release()
        cv2.destroyAllWindows()

node = lane_detect()

node.callback()