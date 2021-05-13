import cv2
import time


def sliding_window(image, stride, windowSize):
    for y in range(0, image.shape[0], stride):
        for x in range(0, image.shape[1], stride):
            yield (x,y, image[ y:y + windowSize[1], x:x + windowSize[0] ])


if __name__ == "__main__" :

    folder_path = "C:\\Users\\z0042t1w\\pavan_ws\\gesture_recognition\\data\\"
    
    image = cv2.imread(folder_path + "palm.PNG")
    winW = 128
    winH = 128
    
    for (x,y,window) in sliding_window(image, stride=128, windowSize=(winW,winH)):
        if window.shape[0] != winH or window.shape[1]!=winW:
            continue
        cv2.rectangle(image, (x,y), (x+ winW, y+winH), (0,255,0), 2)
        cv2.imshow("image", image)
        cv2.waitKey(100)
        time.sleep(0.0025)

