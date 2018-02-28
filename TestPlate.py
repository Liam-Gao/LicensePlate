import cv2
import numpy as np
from Projects.LicensePlate.processingImg import preprocess,BlueHsvImg,findPlateNumberRegion,detect


# angle_flag = False
#
# def find_angle(img, col, row):
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
# #
# #
img = cv2.imread("license4.png")
testImg = img.copy()
gray = cv2.cvtColor(testImg,cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
# # cv2.imshow('binary', binary)
# #
# row,col = binary.shape
# #
# angle = find_angle(binary, col, row)
# if angle_flag:
#     M = cv2.getRotationMatrix2D((col // 2, row // 2), angle / 2, 1)
# else:
#     M = cv2.getRotationMatrix2D((col // 2, row // 2), -angle / 2, 1)
#
# dst = cv2.warpAffine(img, M, (col, row))
# dst_gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
# rett, binaryy = cv2.threshold(dst_gray, 50, 255, cv2.THRESH_BINARY)
# cv2.imshow('dst', dst)
# cv2.imshow('img', binaryy)
#
# M = cv2.getRotationMatrix2D((col // 2, row // 2), -13, 1)
# dst = cv2.warpAffine(img, M, (col, row))
# cv2.imshow('aaddddd', dst)
# ange = detect(dst)
# #
# dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
# # ret, dst_binary = cv2.threshold(dst_gray,50,255,cv2.THRESH_BINARY)
# # cv2.imshow('dsg', dst_binary)
#
# equ = cv2.equalizeHist(dst_gray)
# cv2.imshow('equ',equ)
# gaussian = cv2.GaussianBlur(equ, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
# # 中值滤波
# # median = cv2.medianBlur(gaussian, 5)
# # cv2.imshow('median',median)
# sobel = cv2.Sobel(gaussian, cv2.CV_8U, 1, 0, ksize=3)
# cv2.imshow('sobel',sobel)
# rett, dst_binary = cv2.threshold(sobel, 125, 255, cv2.THRESH_BINARY)
# cv2.imshow('dst_binary',dst_binary)
# element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
# element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
# # 膨胀一次，让轮廓突出
# dilation = cv2.dilate(dst_binary, element2, iterations=1)
# # 腐蚀一次，去掉细节
# erosion = cv2.erode(dilation, element1, iterations=1)
# # 再次膨胀，让轮廓明显一些
# dilation2 = cv2.dilate(erosion, element2, iterations=1)
# cv2.imshow('dilation2', dilation2)
#
# returnimg, contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(dst, contours,-1,(0,255,0),2)
# cv2.imshow('hh', dst)

pts1 = np.float32([[10, 10], [10, 20], [20, 20]])
pts2 = np.float32([[20, 20], [20, 30], [40, 40]])
M = cv2.getAffineTransform(pts1,pts2)
row,col = gray.shape
ddst = cv2.warpAffine(gray, M, (col, row))
cv2.imshow('affine', ddst)
cv2.waitKey(0)
#

