import cv2 as cv
import numpy as np
import time
import os

folder_path = "C:\\Users\\z0042t1w\\pavan_ws\\gesture_recognition\\data\\"

all_images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".PNG")]


for image in all_images:
  img = cv.imread(image)
  #convert the image to HSV
  img_hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
  skin_color_lower_threshold = np.array([0,48,80], dtype = "uint8")
  skin_color_higher_threshold = np.array([20,255,255], dtype = "uint8")
  detect_skin_in_range = cv.inRange(img_hsv, skin_color_lower_threshold, skin_color_higher_threshold)
  img_blurred = cv.blur(detect_skin_in_range, (2,2))
  _,img_thresholded = cv.threshold(img_blurred, 0,255,cv.THRESH_BINARY)
  cv.imshow("thresholded_image",img_thresholded)
  cv.waitKey(1000)

  #contours
  #https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
  contours, hierarchy = cv.findContours(img_thresholded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  contours = max(contours, key=lambda x: cv.contourArea(x))
  cv.drawContours(img, [contours], -1 , [0,255,0],3)
  cv.imshow("contour", img)
  cv.waitKey(1000)


  #convex hull
  #https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
  hull = cv.convexHull(contours)
  #cv.drawContours(img, [hull], -1, [255,0,0],3)
  #cv.imshow("convex_hull", img)

  #convexity defects
  #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
  hull = cv.convexHull(contours, returnPoints=False)
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
      cv.circle(img, far, 4, [0, 0, 255], -1)
  if cnt > 0:
    cnt = cnt+1
  cv.putText(img, str(cnt), (0, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)

  cv.imshow("gesture", img)
  cv.waitKey(10000)