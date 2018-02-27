import cv2
import numpy as np

# 车牌宽度按7.5的比例来分

img = cv2.imread('number_plate1.jpg')

# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower_blue = np.array([105, 43, 46])
# upper_blue = np.array([130, 255, 255])
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
# cv2.imshow('a', mask)
plate = cv2.resize(img, None, fx=2, fy=2)  # 放大两倍
copyplate = plate.copy()
gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
# eqImg = cv2.equalizeHist(gray)
# gaussian = cv2.GaussianBlur(eqImg, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
# 中值滤波
# median = cv2.medianBlur(gaussian, 5)
# sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize=3)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('a', binary)
numyPic = np.array(binary)
row, colum = numyPic.shape
print(row, colum)

# 处理行白点
for x in range(row):
    whiteNum = 0
    white = 255
    black = 0
    flag = 0
    up_down_num = 0
    for y in range(colum):
        # 行白点的数目大于一定值，判断这一行为直线白点,每个字符宽度小于1/6
        if numyPic[x][y] > 125:
            whiteNum += 1

        # 判断跳变次数
        if numyPic[x][y] != flag:
            up_down_num += 1
            flag = numyPic[x][y]

    if whiteNum > colum * 0.55:
        for i in range(colum):
            numyPic[x][i] = 0
    # 黑白跳变小于14次，就说明这行没有字符
    if up_down_num < 14:
        for i in range(colum):
            numyPic[x][i] = 0
# # 扩大图像 是 （列行）
# test = cv2.resize(numyPic, (200,44),interpolation=cv2.INTER_CUBIC)
# returnimg, contours, hierarchy = cv2.findContours(numyPic, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, 0, (0, 255, 0), -1)

cv2.imshow('aaa', numyPic)
edge = cv2.Canny(numyPic, 100, 200, True)


# returnImg, contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# imgg = cv2.drawContours(copyplate, contours, -1, (0,0,255), 1)
#
# for i in range(len(contours)):
#     cnt = contours[i]
#     hull = cv2.convexHull(cnt)
#     area = cv2.minAreaRect(cnt)
#     print("第",i,"个面积：",area)


def scanColumWhitePoint(img, row, colum):
    whitePoint = 0
    for i in range(row):
        if img[i, colum] > 0:
            whitePoint += 1
    return whitePoint


# 根据edge，垂直扫描白点数，没有则记录下最后一次没有白点的位置，记录下白点之后下一次没有白点的位置。
ToBlack = []
ToWhite = []
whitePoint = 0
nextHaveWhitePoint = False
nextHaveBlackPoint = False
for i in range(colum - 1):
    whiteNumber = scanColumWhitePoint(numyPic, row, i)
    # 这里需要修改，把处理过后的图片生成新的图片，处理黑行
    if whiteNumber == 0:
        if scanColumWhitePoint(numyPic, row, i + 1) > 0:
            ToWhite.append(i+1)
    if whiteNumber > 0:
        if scanColumWhitePoint(numyPic, row, i + 1) == 0:
            ToBlack.append(i + 1)

print("ToBlack: ", ToBlack)
# print("ToBlack: ", ToBlack[])
print("ToWhite: ", ToWhite)

for x in range(7):
    cv2.imshow(str(x), numyPic[:, ToWhite[x]:ToBlack[x+1]])

# cv2.imshow('6', numyPic[:,45:53])


# cv2.imshow('aa', numyPic)
# cv2.imshow('a', binary)
# cv2.imshow('aaa', edge)
# cv2.imshow('aafa', imgg)
# element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
# element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
# # 膨胀一次，让轮廓突出
# dilation = cv2.dilate(binhull = cv2.convexHull(cnt)ary, element2, iterations=1)
# # 腐蚀一次，去掉细节
# erosion = cv2.erode(dilation, element1, iterations=1)
# # 再次膨胀，让轮廓明显一些
# dilation2 = cv2.dilate(erosion, element2, iterations=3)

# image, contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# imgg = cv2.drawContours(img, contours, -1, (0,255,255), 1)
# cv2.imshow('a', imgg)
# cv2.imshow('aa', dilation2)

cv2.waitKey(0)
