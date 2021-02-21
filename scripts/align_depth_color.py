import numpy as np
import pyrealsense2 as rs
import cv2 as cv

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("depth scale is:", depth_scale)

clipping_distance_in_meters = 1
clipping_distance = clipping_distance_in_meters / depth_scale

align_to = rs.stream.color
align = rs.align(align_to)

try:

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
    
        if not depth_frame or not color_frame:
            continue

        img_depth = np.asarray(depth_frame.get_data())
        img_color = np.asarray(color_frame.get_data())
        grey_color = 50
        img_depth_3d = np.dstack((img_depth,img_depth,img_depth))
        bg_removed = np.where((img_depth_3d > clipping_distance) | (img_depth_3d <=0.5), grey_color, img_color)

        colormap_depth = cv.applyColorMap(cv.convertScaleAbs(img_depth, alpha=0.03),cv.COLORMAP_JET)
        imgaes = np.hstack((bg_removed, colormap_depth))

        cv.namedWindow("opencv_viewer", cv.WINDOW_AUTOSIZE)
        cv.imshow("opencv_viewer", imgaes)
        key = cv.waitKey(1)

        if key & 0xFF == ord('q') or key == 27 :
            cv.destroyAllWindows()
            break
finally:
    pipeline.stop()

