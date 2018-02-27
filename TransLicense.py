import cv2
import numpy as np
from Projects.LicensePlate.processingImg import preprocess,BlueHsvImg
#
angle_flag = False


def find_angle(img, col, row):
    left = col * 0.25
    mid = col * 0.5
    right = col * 0.75
    a = b = 0
    while img[a][int(left)] == 255:
        a += 1
    while img[b][int(right)] == 255:
        b += 1
    if a-b < 0:
        angle_flag = True
    else:
        angle_flag = False
    theta = np.arctan(abs(a-b)/(right - left))
    print(theta)
    angle = theta * 360 // np.pi
    return int(angle)


img = cv2.imread("number_plate1.jpg")
testImg = img.copy()


gray = cv2.cvtColor(testImg,cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
cv2.imshow('binary', binary)

row,col = binary.shape

angle = find_angle(binary, col, row)
if angle_flag:
    M = cv2.getRotationMatrix2D((col // 2, row // 2), angle / 2, 1)
else:
    M = cv2.getRotationMatrix2D((col // 2, row // 2), -angle / 2, 1)

bigger = cv2.copyMakeBorder(img, row//4,row//4,col//4,col//4,cv2.BORDER_CONSTANT,value=(0,0,0))
brow,bcol = bigger.shape[:2]

dst = cv2.warpAffine(bigger, M, (bcol, brow))
# dst_gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
# rett, binaryy = cv2.threshold(dst_gray, 50, 255, cv2.THRESH_BINARY)

cv2.imshow('dst', dst)
cv2.imshow('bigger', bigger)
dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
di = preprocess(dst_gray)
hsb = BlueHsvImg(dst)
cv2.imshow('ddddd', di)

cv2.waitKey(0)