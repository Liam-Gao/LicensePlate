import cv2
import numpy as np

np.set_printoptions(threshold=np.inf)

img = cv2.imread('license4.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)
cv2.imshow('hhh', s)

ret, binary = cv2.threshold(s, 100, 255, cv2.THRESH_BINARY)
cv2.imshow('bi', binary)

element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
dilation = cv2.dilate(binary, element2, iterations=1)
erosion = cv2.erode(dilation, element1, iterations=1)
cv2.imshow('di', dilation)
cv2.imshow('er', erosion)

returnimg, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
drawing = cv2.drawContours(img, contours, -1, (0,255,0), 3)
cv2.imshow('drwa', drawing)

# print("cont: ", contours)

for i in range(len(contours)):
    cnt = contours[i]
    area = cv2.contourArea(cnt)
    if area < 2000:
        continue
    # print("areas: ", area)

    rect = cv2.minAreaRect(cnt)
    print("rect: ", rect)

    box = cv2.boxPoints(rect)
    print("box: ", box)

    width = abs(box[0][0] - box[2][0])
    height = abs(box[0][1] - box[2][1])

    ration = float(width)/float(height)
    print("ration", ration)
    if ration < 1.0 or ration > 5.0:
        continue
    # print("now: ", cnt)


cv2.waitKey(0)