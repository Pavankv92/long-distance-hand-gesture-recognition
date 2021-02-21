import numpy as np
import pyrealsense2 as rs
import cv2 as cv
import datetime
import os


class GestureRecognition(object):

    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.skin_color_lower_threshold = np.array([0,48,80], dtype = "uint8")
        self.skin_color_higher_threshold = np.array([20,255,255], dtype = "uint8")
        self.max_distance_to_hand = 2.0
        self.min_distance_to_hand = 0.5
        self.stratify_depth = 0.1
        print("depth scale is:", self.depth_scale)

    def get_clipping_distance(self, clipping_distance_in_meters):
        return clipping_distance_in_meters/self.depth_scale

    def display_image(self, img):
        cv.namedWindow("opencv_viewer", cv.WINDOW_AUTOSIZE)
        cv.imshow("opencv_viewer", img)
        cv.waitKey(10000)
    def save_image(self, img):
        filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join("C:\\Users\\z0042t1w\\pavan_ws\\gesture_recognition\\data\\depth_stratification_imgs\\", str(filename))
        cv.imwrite("{name}.png".format(name = str(path)), img)

    def main(self):
        try:
            found_image = False
            while not found_image:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
            
                if not depth_frame or not color_frame:
                    continue

                img_depth = np.asarray(depth_frame.get_data())
                img_color = np.asarray(color_frame.get_data())
                grey_color = 50
                img_depth_3d = np.dstack((img_depth,img_depth,img_depth))
                #self.stratify_depth(img_color, img_depth_3d, img_depth, grey_color)
                depth_array = np.arange(self.min_distance_to_hand, self.max_distance_to_hand, self.stratify_depth)
                image_list = []
                for k in range(len(depth_array)):
                    if k == 0 :
                        continue    
                    bg_removed = np.where((img_depth_3d > self.get_clipping_distance(depth_array[k])) | (img_depth_3d <= self.get_clipping_distance(depth_array[k-1])), grey_color, img_color)
                    image_list.append(bg_removed)
                
                for img in image_list :
                    img_hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
                    detect_skin_in_range = cv.inRange(img_hsv, self.skin_color_lower_threshold, self.skin_color_higher_threshold)
                    img_blurred = cv.blur(detect_skin_in_range, (2,2))
                    _,img_thresholded = cv.threshold(img_blurred, 0,255, cv.THRESH_BINARY)

                    #self.display_image(img_thresholded)
                    self.save_image(img_thresholded)
                    try:
                        #print("trying contour now")
                        contours, hierarchy = cv.findContours(img_thresholded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                        contours = max(contours, key=lambda x: cv.contourArea(x))
                    
                        hull = cv.convexHull(contours, returnPoints=False)

                        cv.drawContours(img_color, [hull], -1, [255,0,0],3)
                        cv.imshow("convex_hull", img_color)
                        defects = cv.convexityDefects(contours, hull)

                        if defects is not None:
                            cnt = 0
                        for i in range(defects.shape[0]):  # calculate the angle
                            s, e, f, d = defects[i][0]
                            start = tuple(contours[s][0])
                            end = tuple(contours[e][0])
                            far = tuple(contours[f][0])
                            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  #      cosine theorem
                            if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
                                cnt += 1
                                cv.circle(img_color, far, 4, [0, 0, 255], -1)
                        if cnt > 0:
                            cnt = cnt+1
                        cv.putText(img_color, str(cnt), (0, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)
                        self.display_image(img_color)

                    except Exception:
                        continue
                    
                    """
                    cv.namedWindow("opencv_viewer", cv.WINDOW_AUTOSIZE)
                    cv.imshow("opencv_viewer", img_thresholded)
                    key = cv.waitKey(1000)
                    if key & 0xFF == ord('q') or key == 27 :
                       
                        cv.destroyAllWindows()
                    """     
        finally:
            self.pipeline.stop()


if __name__ == "__main__":
    gr = GestureRecognition()
    gr.main()

