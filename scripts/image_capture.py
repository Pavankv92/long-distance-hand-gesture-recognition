import numpy as np
import pyrealsense2 as rs
import cv2 as cv
import datetime
import os

pipeline = rs.pipeline()
config = rs.config()
#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)


try:
    response = " "
    while not response == "q":
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()

        if not color:
            continue
        img_color = np.asarray(color.get_data())
        
        cv.namedWindow("opencv_viewer", cv.WINDOW_AUTOSIZE)
        cv.imshow("opencv_viewer", img_color)
        filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join("C:\\Users\\z0042t1w\\pavan_ws\\gesture_recognition\\data\\", str(filename))
        cv.imwrite("{name}.png".format(name = str(path)), img_color)
        #cv.imwrite(path, img_color)
        cv.waitKey(500)
        cv.destroyAllWindows()
        response = input("image saved, change the gesture and hit enter or press q to quit")
finally:
    
    pipeline.stop()



