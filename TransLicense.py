import cv2
import numpy as np
from Projects.LicensePlate.processingImg import detect

# imggg = cv2.imread("license4.png")
# # imggg = cv2.imread("timg.jpg")
# angle, angle_flag = detect(imggg)
#
# if angle < 0 and angle > -75:
#     row, col = imggg.shape[:2]
#     if angle_flag:
#         M = cv2.getRotationMatrix2D((col // 2, row // 2), angle, 1)
#     else:
#         M = cv2.getRotationMatrix2D((col // 2, row // 2), -angle, 1)
#
#     dst = cv2.warpAffine(imggg, M, (col, row))
#     angle, angle_flag = detect(dst)

img = cv2.imread('number_plate1.jpg')
cv2.imshow('img', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
ret, binary = cv2.threshold(gaussian, 50, 255, cv2.THRESH_BINARY)
can = cv2.Canny(binary, 100, 200)
cv2.imshow('can', can)
cv2.imshow('binary', binary)
#
newSize = cv2.resize(can, (440, 140))
newImg = cv2.resize(img, (440, 140))
rows,cols = newImg.shape[:2]
cv2.imshow('newSize', newSize)


# 判断倾斜方向和计算车牌倾斜角度


def caculateRotation(img):
    row, col = img.shape
    #     取1/4和3/4行比较大小，上方大 return True
    count14 = 0
    count34 = 0
    row14 = int(row * 0.25)
    row34 = int(row * 0.75)
    for i in range(col):
        if img[row14][i] == 0:
            count14 += 1
        else:
            break
    for i in range(col):
        if img[row34][i] == 0:
            count34 += 1
        else:
            break
    print(count14, count34)
    if count14 > count34:
        return True
    else:
        return False


def caculateAngle(img):
    # 取（1/4和1/2）和（1/2和3/4）斜率的平均值
    row, col = img.shape
    x14 = x12 = x34 = y14 = y12 = y34 = 0
    for y in range(col):
        if img[int(row * 0.25)][y] == 0:
            x14 += 1
        else:
            y14 = int(row * 0.25)
            break
    for y in range(col):
        if img[int(row * 0.5)][y] == 0:
            x12 += 1
        else:
            y12 = int(row * 0.5)
            break
    for y in range(col):
        if img[int(row * 0.75)][y] == 0:
            x34 += 1
        else:
            y34 = int(row * 0.75)
            break
    angle1412 = np.arctan(abs(x14 - x12) / abs(y14 - y12)) * 360 // np.pi
    angle1234 = np.arctan(abs(x12 - x34) / abs(y12 - y34)) * 360 // np.pi
    angle1434 = np.arctan(abs(x14 - x34) / abs(y14 - y34)) * 360 // np.pi
    angle = (angle1412 + angle1234 + angle1434) // 3
    return angle, x14, x12, x34, y14, y12, y34


flag = caculateRotation(newSize)
angle, x1, x2, x3, y1, y2, y3 = caculateAngle(newSize)
print(x1,x2,x3,y1,y2,y3)
# 进行仿射变换
if flag:
    pts1 = np.float32([[np.tan(angle)*y1, 0], [x2, y2], [440, 0]])
    pts2 = np.float32([[0, 0], [x2, y2], [440-np.tan(angle)*140, 0]])
    M = cv2.getAffineTransform(pts1,pts2)
    ddst = cv2.warpAffine(newImg, M, (cols, rows))
    cv2.imshow('affine', ddst)
    cv2.imwrite('thistest.jpg',ddst)
cv2.waitKey()
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
