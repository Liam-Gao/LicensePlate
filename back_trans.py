import cv2
import numpy as np
from Projects.LicensePlate.back_process import detect


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



def removeLR_Black(binary):
    for col in range(binary.shape[1]):
        count = 0
        for row in range(binary.shape[0]):
            if binary[row, col] > 100:
                count += 1
        if count > 3:
            colStart = col
            break
    # 结束列
    for col in range(binary.shape[1], -1, -1):
        count = 0
        for row in range(binary.shape[0]):
            if binary[row, col - 1] > 100:
                count += 1
        if count > 3:
            colEnd = col
            break
    roi = binary[:, colStart:colEnd]
    return roi


def removeUD_Black(binary):
    picRow, picCol = binary.shape
    for x in range(picRow):
        lengthList = []
        lenStart = 0
        lenEnd = 0
        countWhitePoints = 0
        for y in range(picCol - 1):
            if binary[x][y] > 0 and binary[x][y + 1] > 0:
                lenStart = y
                lengthList.append(lenStart)
            if binary[x][y] > 0:
                countWhitePoints += 1
        # 如果跳变次数小于10即lengthList的长度小于10，则把这一行统统置为0
        # print(countWhitePoints)
        if lengthList.__len__() < 10 or countWhitePoints < 50:
            for y in range(picCol):
                binary[x][y] = 0

    list = []
    for x in range(picRow):
        whitePoints = []
        newLenStart = 0
        for y in range(picCol - 1):
            if binary[x][y] > 0 and binary[x][y + 1] > 0:
                newLenStart = y
                whitePoints.append(newLenStart)

        list.append(whitePoints)
    for x in range(len(list) - 1):
        # 第一行的判断
        if x == 0:
            if list[x + 1] == [] and list[x + 2] == [] and len(list[x]) != 0:
                for y in range(picCol):
                    binary[x][y] = 0
        # 最后一行的判断
        elif x == len(list):
            if list[x - 1] == [] and list[x - 2] == [] and len(list[x]) != 0:
                for y in range(picCol):
                    binary[x][y] = 0
        # 中间行的判断
        else:
            if list[x - 1] == [] and list[x + 1] == [] and len(list[x]) != 0:
                for y in range(picCol):
                    binary[x][y] = 0
    return binary


def hsvProcess(img, binary):
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsvimg', hsvimg)
    # h, s, v = cv2.split(hsvimg)
    # hh = cv2.equalizeHist(h)
    # ss = cv2.equalizeHist(s)
    # vv = cv2.equalizeHist(v)
    # newhsv = cv2.merge([hh, ss, vv])
    # cv2.imshow('newhsv', newhsv)
    lower_blue = np.array([80, 43, 46])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsvimg, lower_blue, upper_blue)
    # cv2.imshow('mask', mask)
    mm = cv2.bitwise_and(mask, binary)
    cv2.imshow('hsv&binary', mm)
    return mm

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
# cv2.imshow('img', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
ret, binary = cv2.threshold(gaussian, 50, 255, cv2.THRESH_BINARY)
can = cv2.Canny(binary, 100, 200)
# cv2.imshow('can', can)
cv2.imshow('binary', binary)
#
newSize = cv2.resize(can, (440, 140))
newImg = cv2.resize(img, (440, 140))
rows,cols = newImg.shape[:2]
cv2.imshow('newSize', newSize)

# ------------------------------------------------------------------------------------

flag = caculateRotation(newSize)
angle, x1, x2, x3, y1, y2, y3 = caculateAngle(newSize)
print(x1,x2,x3,y1,y2,y3)
# 进行仿射变换
pts1 = None
pts2 = None
if flag:
    pts1 = np.float32([[np.tan(angle)*y1, 0], [x2, y2], [440, 0]])
    pts2 = np.float32([[0, 0], [x2, y2], [440-np.tan(angle)*140, 0]])
else:
    pts1 = np.float32([[np.tan(angle) * 140, 140], [x2, y2], [440, 140]])
    pts2 = np.float32([[0, 140], [x2, y2], [440-np.arctan(angle)*140, 140]])

M = cv2.getAffineTransform(pts1, pts2)
ddst = cv2.warpAffine(newImg, M, (cols, rows))
cv2.imshow('affine', ddst)
# ---------------------------------------------------------------------------------------
cut = cv2.imread('thistest.jpg')


cutt = cut.copy()
gg = cv2.cvtColor(cut,cv2.COLOR_BGR2GRAY)
r,b = cv2.threshold(gg,50,255,cv2.THRESH_BINARY)

hsv_binary = hsvProcess(cut, b)

newBinary = removeLR_Black(hsv_binary)
newBinaryy = removeUD_Black(newBinary)
cv2.imshow('removeLR', newBinary)
cv2.imshow('removeUD', newBinaryy)


reImg, contours, hierarchy = cv2.findContours(newBinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
picRow, picCol = cutt.shape[:2]
picPoints = []
for contour in contours:
    if cv2.contourArea(contour) < 1000:
        continue
    x, y, w, h = cv2.boundingRect(contour)
    if h < picRow / 2.5:
        continue
    if w < 8 or w > picCol * 1/4:
        continue
    print(x,y,w,h)
    picPoints.append([x,y,w,h])
for xx, yy, ww, hh in picPoints:
    testPic = cv2.rectangle(cutt, (xx, yy ), (xx + ww , yy + hh ), (0, 255, 0), 1)
cv2.imshow('aaaa',cutt)
cv2.imshow('bbbb', b)



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