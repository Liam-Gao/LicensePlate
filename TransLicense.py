import cv2
import numpy as np
from Projects.LicensePlate.processingImg import preprocess,BlueHsvImg,findPlateNumberRegion,detect

imggg = cv2.imread("license4.png")
# imggg = cv2.imread("timg.jpg")
angle, angle_flag = detect(imggg)

if angle < 0 and angle > -75:
    row, col = imggg.shape[:2]
    if angle_flag:
        M = cv2.getRotationMatrix2D((col // 2, row // 2), angle, 1)
    else:
        M = cv2.getRotationMatrix2D((col // 2, row // 2), -angle, 1)

    dst = cv2.warpAffine(imggg, M, (col, row))
    angle, angle_flag = detect(dst)

# def isRotated(binary):
# #     往左上旋转为True
#     row,col = binary.shape
#
#     return True

# def find_angle(img, col, row):
#     global angle_flag
#     left = col * 0.25
#     mid = col * 0.5
#     right = col * 0.75
#     a = b = 0
#     while img[a][int(left)] == 255:
#         a += 1
#     while img[b][int(right)] == 255:
#         b += 1
#     if a-b < 0:
#         angle_flag = True
#     else:
#         angle_flag = False
#     theta = np.arctan(abs(a-b)/(right - left))
#     print(theta)
#     angle = theta * 360 // np.pi
#     return int(angle)


# imggg = cv2.imread("license4.png")
# angle = detect(imggg)
# img = cv2.imread("number_plate1.jpg")
# testImg = img.copy()
#
#
# gray = cv2.cvtColor(testImg,cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
# cv2.imshow('binary', binary)
#
# row,col = binary.shape
# angle_flag = isRotated(binary)
#
# if angle_flag:
#     M = cv2.getRotationMatrix2D((col // 2, row // 2), angle, 1)
# else:
#     M = cv2.getRotationMatrix2D((col // 2, row // 2), -angle, 1)
# #
# bigger = cv2.copyMakeBorder(img, row//4,row//4,col//4,col//4,cv2.BORDER_CONSTANT,value=(0,0,0))
# brow,bcol = bigger.shape[:2]
# #
# dst = cv2.warpAffine(bigger, M, (bcol, brow))
# dst_gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
# rett, binaryy = cv2.threshold(dst_gray, 50, 255, cv2.THRESH_BINARY)
#
# cv2.imshow('dst', dst)
# cv2.imshow('bigger', bigger)
# dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
#
# dilation = preprocess(dst_gray)
# hsvBinary = BlueHsvImg(dst)
#
# cv2.imshow('ddddd', dilation)
# getCombImg = cv2.bitwise_and(dilation, hsvBinary)
# cv2.imshow('combbb', getCombImg)
#

cv2.waitKey(0)