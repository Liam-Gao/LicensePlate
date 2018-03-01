import cv2
import numpy as np
import operator
import math

# def returnWhitePoints(img):
#     testdata = []
#     for x in range(img.shape[0]):
#         whiteNumRow = 0
#         for y in range(img.shape[1]):
#             if int(img[x][y]) > 175:
#                 whiteNumRow += 1
#         testdata.append(whiteNumRow)
#         #     列白点
#     for x in range(img.shape[1]):
#         whiteNumCol = 0
#         for y in range(img.shape[0]):
#             if int(img[y][x]) > 175:
#                 whiteNumCol += 1
#         testdata.append(whiteNumCol)
#     return testdata
#
#
# # 欧氏距离
# def eulideanDistance(testList, trainList, value):
#     distance = 0
#     for x in range(len(trainList)):
#         distance += pow((testList[x] - trainList[x]), 2)
#     distances = math.sqrt(distance)
#     return [distances, value]


np.set_printoptions(threshold=np.inf)

# img = cv2.imread('number_plate1.jpg')
img = cv2.imread('thistest.jpg')

# cv2.imshow('original', img)

# HSV处理
hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow('hsvimg', hsvimg)

h, s, v = cv2.split(hsvimg)
hh = cv2.equalizeHist(h)
ss = cv2.equalizeHist(s)
vv = cv2.equalizeHist(v)
newhsv = cv2.merge([hh, ss, vv])
# cv2.imshow('s', s)

lower_blue = np.array([80, 43, 46])
upper_blue = np.array([140, 255, 255])
mask = cv2.inRange(newhsv, lower_blue, upper_blue)
cv2.imshow('mask', mask)

print(mask.shape)
colStart = 0
colEnd = 0

# 再次扫描，去掉车牌左右黑色部分
# 起始列
for col in range(mask.shape[1]):
    count = 0
    for row in range(mask.shape[0]):
        if mask[row, col] > 100:
            count += 1
    if count > 3:
        colStart = col
        break
# 结束列
for col in range(mask.shape[1], -1, -1):
    count = 0
    for row in range(mask.shape[0]):
        if mask[row, col - 1] > 100:
            count += 1
    if count > 3:
        colEnd = col
        break
print("起始与结束: ", colStart, colEnd)

roi = img[:, colStart:colEnd]

cv2.imshow("ROI区域", roi)

newRoi = cv2.resize(roi, (440, 140))
# cv2.imshow("NewROI区域", newRoi)

# ---------形态学处理---------------------------------
gray = cv2.cvtColor(newRoi, cv2.COLOR_BGR2GRAY)
# print(gray)
# equ = cv2.equalizeHist(gray)
# gauss = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
# median = cv2.medianBlur(gauss, 5)
# cv2.imshow('media', median)
# Sobel算子，X方向求梯度
# sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
# cv2.imshow('sobel', sobel)
# 二值化
ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
# cv2.imshow('binary', binary)

# --------进行扫描，去掉上下边框-----------------------------------
print("ROI shape:", newRoi.shape)
picRow = newRoi.shape[0]
picCol = newRoi.shape[1]

# 竖着切下去
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

cv2.imshow("process binary: ", binary)


# 横着切过去，上一行 下一行的数值几乎都是黑色，那这一行就归为黑色
list = []
for x in range(picRow):
    whitePoints = []
    newLenStart = 0
    for y in range(picCol - 1):
        if binary[x][y] > 0 and binary[x][y + 1] > 0:
            newLenStart = y
            whitePoints.append(newLenStart)
            # print(whitePoints)
            # if binary[x][y] > 0:
            #     whitePoints.append(y)
    list.append(whitePoints)
print('list1',list[1])
# print(len(list[0]))
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

cv2.imshow("last process binary: ", binary)
testPic = newRoi.copy()
reImg, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
picPoints = []
# widthList = []
for contour in contours:
    if cv2.contourArea(contour) < 300:
        continue
    # print(contour)
    x, y, w, h = cv2.boundingRect(contour)
    if h < picRow / 2.5:
        continue
    if w < 8:
        continue
    picPoints.append([x, y, w, h])

print(picPoints)
picPoints.sort(key=operator.itemgetter(0))
print(picPoints)

# 点不包含汉字在内
if len(picPoints) == 7:
    picPoints.remove(picPoints[0])
# print(picPoints)

x_points_start = []
x_points_end = []
y_start = picPoints[0][1] - 5
y_end = picPoints[0][1] + picPoints[0][3] + 5

for xx, yy, ww, hh in picPoints:
    testPic = cv2.rectangle(testPic, (xx - 5, yy - 5), (xx + ww + 5, yy + hh + 5), (0, 255, 0), 2)
    x_points_start.append(xx - 5)
    x_points_end.append(xx + ww + 5)
print(x_points_start)
print(x_points_end)

firstWidth = x_points_end[0] - x_points_start[0]

firstX = x_points_start[0] - firstWidth
firstX_End = x_points_start[0]
print(firstX, firstX_End)

chinese_pic = None

# 判断汉字左边和上边越界问题
if firstX < 0:
    if y_start < 0:
        cv2.imshow("FirstNumber", binary[0:y_end, 0:firstX_End])
        chinese_pic = binary[0:y_end, 0:firstX_End].copy()
    else:
        cv2.imshow("FirstNumber", binary[y_start:y_end, 0:firstX_End])
        chinese_pic = binary[y_start:y_end, 0:firstX_End].copy()
else:
    if y_start < 0:
        cv2.imshow("FirstNumber", binary[0:y_end, firstX:firstX_End])
        chinese_pic = binary[0:y_end, firstX:firstX_End].copy()
    else:
        cv2.imshow("FirstNumber", binary[y_start:y_end, firstX:firstX_End])
        chinese_pic = binary[y_start:y_end, firstX:firstX_End].copy()

cv2.imshow('test', testPic)
cv2.imshow('chinese', chinese_pic)
cv2.imwrite('p0.jpg', chinese_pic)

#
for i in range(6):
    if y_start < 0:
        cv2.imshow("第" + str(i) + "个", binary[0:y_end, x_points_start[i]:x_points_end[i]])
        cv2.imwrite('p'+str(i+1)+'.jpg', binary[0:y_end, x_points_start[i]:x_points_end[i]])
    else:
        cv2.imshow("第" + str(i) + "个", binary[y_start:y_end, x_points_start[i]:x_points_end[i]])
        cv2.imwrite('p' + str(i + 1) + '.jpg', binary[0:y_end, x_points_start[i]:x_points_end[i]])


cv2.waitKey(0)


