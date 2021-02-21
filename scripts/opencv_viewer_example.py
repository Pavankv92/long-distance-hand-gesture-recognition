import numpy as np
import pyrealsense2 as rs
import cv2 as cv

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)


try:

    while True:
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()

        if not depth or not color:
            continue
        
        img_depth = np.asarray(depth.get_data())
        img_color = np.asarray(color.get_data())
        colormap_depth = cv.applyColorMap(cv.convertScaleAbs(img_depth, alpha=0.03), cv.COLORMAP_JET)
        imgaes = np.hstack((img_color, colormap_depth))
        cv.namedWindow("opencv_viewer", cv.WINDOW_AUTOSIZE)
        cv.imshow("opencv_viewer", imgaes)
        cv.waitKey(0)
finally:
    
    pipeline.stop()


