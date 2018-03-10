import cv2
import numpy as np
import operator
from Projects.LicensePlate.processingImg import detect
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


# np.set_printoptions(threshold=np.inf)


def countLenth(row, col, mask):
    count = 0
    for i in range(col):
        if mask[row][i] != 255:
            count += 1
        else:
            break
    return count


def hsv_binary_process(img):
    # HSV处理
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # # cv2.imshow('hsvimg', hsvimg)
    #
    # h, s, v = cv2.split(hsvimg)
    # # hh = cv2.equalizeHist(h)
    # # ss = cv2.equalizeHist(s)
    # # vv = cv2.equalizeHist(v)
    # # newhsv = cv2.merge([hh, ss, vv])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    re, binr = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    #
    # cv2.imshow('binr', binr)
    # cv2.waitKey(0)
    lower_blue = np.array([80, 43, 46])
    upper_blue = np.array([115, 255, 255])
    mask = cv2.inRange(hsvimg, lower_blue, upper_blue)
    newIm = cv2.bitwise_and(binr, mask)
    return newIm


def findSlope(mask):
    row, col = mask.shape
    row1 = int(row * 0.25)
    row2 = int(row * 0.5)
    row3 = int(row * 0.75)
    len1 = countLenth(row1, col, mask)
    len2 = countLenth(row2, col, mask)
    len3 = countLenth(row3, col, mask)
    slope1 = (len3 - len1) / (row3 - row1)
    slope2 = (len3 - len2) / (row3 - row2)
    slope3 = (len2 - len1) / (row2 - row1)
    if slope1 < slope2 and slope1 < slope3:
        return slope1
    elif slope2 < slope1 and slope2 < slope3:
        return slope2
    else:
        return slope3
    # slope = (slope1 + slope2 + slope3) / 3
    # print('slope',slope3,slope2,slope1)
    # return slope


def calculateAngle(binary):
    pass


def cutNoAnglePlate(angle, angle_flag, oriImg):
    img = cv2.imread("cutplate.jpg")
    # rows, cols = img.shape[:2]
    # img = cv2.imread('number_plate1.jpg')
    # img = cv2.imread('thistest.jpg')

    # cv2.imshow('original', img)

    # # HSV处理
    # mask = hsv_binary_process(img)


    # cv2.imshow('mask', newIm)
    # cv2.waitKey(0)

    # print(mask.shape)
    # colStart = 0
    # colEnd = 0

    # 再次扫描，去掉车牌左右黑色部分
    # 起始列
    # for col in range(mask.shape[1]):
    #     count = 0
    #     for row in range(mask.shape[0]):
    #         if mask[row, col] > 100:
    #             count += 1
    #     if count > 3:
    #         colStart = col
    #         break
    # # 结束列
    # for col in range(mask.shape[1], -1, -1):
    #     count = 0
    #     for row in range(mask.shape[0]):
    #         if mask[row, col - 1] > 100:
    #             count += 1
    #     if count > 3:
    #         colEnd = col
    #         break
    # print("起始与结束: ", colStart, colEnd)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # roi = img[:, colStart:colEnd]
    #
    # cv2.imshow("ROI区域", roi)
    # cv2.waitKey(0)

    # ------------------------旋转图片------------------------------------------------------------------
    # grayi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # binary_roi = cv2.threshold(grayi, 100, 255, cv2.THRESH_BINARY)
    row, col = img.shape[:2]
    if angle < 0 and angle > -80:
        if angle_flag:
            M = cv2.getRotationMatrix2D((col // 2, row // 2), angle, 1)
        else:
            M = cv2.getRotationMatrix2D((col // 2, row // 2), -angle, 1)

        dst = cv2.warpAffine(img, M, (col, row))
        cv2.imshow('dst', dst)
        cv2.waitKey()
    else:
        dst = img

    # hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    # lower_blue = np.array([80, 43, 46])
    # upper_blue = np.array([115, 255, 255])
    # mas = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = hsv_binary_process(dst)
    cv2.imshow('mask', mask)
    # 去掉上下杂质
    for i in range(row):
        count = 0
        for j in range(col):
            if mask[i][j] != 0:
                count += 1
        if count < col /10:
            for q in range(col):
                mask[i][q] = 0

    cv2.imshow('mask2', mask)

    upCount = downCount = 0
    for i in range(row):
        count = 0
        for j in range(col):
            if mask[i][j] != 0:
                count += 1
        if count < col / 10:
            continue
        else:
            upCount = i
            break
    for i in range(row - 1, 0, -1):
        count = 0
        for j in range(col):
            if mask[i][j] != 0:
                count += 1
        if count < col / 10:
            continue
        else:
            downCount = i
            break
    # newImg = dst[upCount:downCount, :]
    # newRows, newCols = newImg.shape[:2]
    # cv2.imshow('newiIIII', newImg)
    # cv2.waitKey()
    # detect(dst)
    # imgg = cv2.imread("cutplate.jpg")
    # newImgg = cv2.resize(imgg, (440, 140))
    # maskk = hsv_binary_process(newImgg)
    # newImg = cv2.copyMakeBorder(newImgg, rows//4, rows//4, cols//4, cols//4, cv2.BORDER_CONSTANT, value=(0,0,0))
    # reImgg = cv2.resize(newImg, (440, 140))
    # grayy = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)
    # gass = cv2.GaussianBlur(grayy, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    # rr, bb = cv2.threshold(gass, 100, 255, cv2.THRESH_BINARY)
    # cv2.imshow('binarrr', bb)

    # 去除左右黑边
    leftStart = rightStart = 0
    for i in range(col):
        count = 0
        for j in range(row):
            if mask[j][i] != 0:
                count += 1
        if count != 0:
            leftStart = i
            break
    for i in range(col-1, 0, -1):
        count = 0
        for j in range(row):
            if mask[j][i] != 0:
                count += 1
        if count != 0:
            rightStart = i
            break

    newImg = dst[upCount:downCount, leftStart:rightStart]
    cv2.imshow('newImg', newImg)

    grayy = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)
    gass = cv2.GaussianBlur(grayy, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    rr, bb = cv2.threshold(gass, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow('bb', bb)

    newRow, newCol = bb.shape
    print('height', newRow, 'widt', newCol)
    cv2.waitKey()
    slope = findSlope(mask)

    xiddf = int(newRow * abs(slope))
    print('xiddf', xiddf)
    if slope < 0:
        pts1 = np.float32([[0, newRow-1], [xiddf, 0], [newCol-1, 0]])
        pts2 = np.float32([[xiddf/2, newRow-1], [xiddf/2, 0], [newCol-1-xiddf/2, 0]])
    else:
        pts1 = np.float32([[0, newRow-1], [newCol-1, newRow-1], [newCol - 1 - xiddf, 0]])
        pts2 = np.float32([[0, newRow-1], [newCol-1, newRow-1], [newCol - 1, 0]])

        print('123123',pts1, pts2)
    M = cv2.getAffineTransform(pts1, pts2)
    ddst = cv2.warpAffine(bb, M, (newCol, newRow))
    cv2.imshow('affine', ddst)
    cv2.waitKey()



    # else:
    #     newImg = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=BLUE)
    #     slope = findSlope(mask)
        #     ---------------------------------给图像扩边----------------------------------------

    # newRoi = cv2.resize(roi, (440, 140))
    # cv2.imshow("NewROI区域", newRoi)

    # ---------形态学处理---------------------------------
    # gray = cv2.cvtColor(newRoi, cv2.COLOR_BGR2GRAY)
    # print(gray)
    # equ = cv2.equalizeHist(gray)
    # gauss = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    # median = cv2.medianBlur(gauss, 5)
    # cv2.imshow('media', median)
    # Sobel算子，X方向求梯度
    # sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # cv2.imshow('sobel', sobel)
    # 二值化
    # ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    # cv2.imshow('binary', binary)
    # can = cv2.Canny(binary, 100, 200)
    # cv2.imshow('can', can)


    # angle = calculateAngle(binary)
    # cv2.waitKey()
    #
    #
    # # --------进行扫描，去掉上下边框-----------------------------------
    # print("ROI shape:", ddst.shape)
    # picRow = ddst.shape[0]
    # picCol = ddst.shape[1]
    #
    # # 竖着切下去
    # for x in range(picRow):
    #     lengthList = []
    #     lenStart = 0
    #     lenEnd = 0
    #     countWhitePoints = 0
    #     for y in range(picCol - 1):
    #         if ddst[x][y] > 0 and ddst[x][y + 1] > 0:
    #             lenStart = y
    #             lengthList.append(lenStart)
    #         if ddst[x][y] > 0:
    #             countWhitePoints += 1
    #     # 如果跳变次数小于10即lengthList的长度小于10，则把这一行统统置为0
    #     # print(countWhitePoints)
    #     if lengthList.__len__() < 10 or countWhitePoints < 50:
    #         for y in range(picCol):
    #             ddst[x][y] = 0
    #
    # cv2.imshow("process binary: ", ddst)
    # cv2.waitKey()
    #
    # # 横着切过去，上一行 下一行的数值几乎都是黑色，那这一行就归为黑色
    # list = []
    # for x in range(picRow):
    #     whitePoints = []
    #     newLenStart = 0
    #     for y in range(picCol - 1):
    #         if ddst[x][y] > 0 and ddst[x][y + 1] > 0:
    #             newLenStart = y
    #             whitePoints.append(newLenStart)
    #             # print(whitePoints)
    #             # if binary[x][y] > 0:
    #             #     whitePoints.append(y)
    #     list.append(whitePoints)
    # print('list1',list[1])
    # # print(len(list[0]))
    # for x in range(len(list) - 1):
    #     # 第一行的判断
    #     if x == 0:
    #         if list[x + 1] == [] and list[x + 2] == [] and len(list[x]) != 0:
    #             for y in range(picCol):
    #                 ddst[x][y] = 0
    #     # 最后一行的判断
    #     elif x == len(list):
    #         if list[x - 1] == [] and list[x - 2] == [] and len(list[x]) != 0:
    #             for y in range(picCol):
    #                 ddst[x][y] = 0
    #     # 中间行的判断
    #     else:
    #         if list[x - 1] == [] and list[x + 1] == [] and len(list[x]) != 0:
    #             for y in range(picCol):
    #                 ddst[x][y] = 0
    #
    # cv2.imshow("last process binary: ", ddst)
    # testPic = ddst.copy()
    # reImg, contours, hierarchy = cv2.findContours(testPic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # picPoints = []
    # # widthList = []
    # for contour in contours:
    #     if cv2.contourArea(contour) < 300:
    #         continue
    #     # print(contour)
    #     x, y, w, h = cv2.boundingRect(contour)
    #     if h < picRow / 2.5:
    #         continue
    #     if w < 20:
    #         continue
    #     picPoints.append([x, y, w, h])
    #
    # picPoints.sort(key=operator.itemgetter(0))
    #
    # print('picponits: ', picPoints)
    # # 点不包含汉字在内
    # # print('-------------', len(picPoints))
    # # if len(picPoints) == 7:
    # #     picPoints.remove(picPoints[0])
    # # print(picPoints)
    # if len(picPoints) == 6:
    #     chinese_x_end = picPoints[0][0] - 7
    #     chinese_y_point = picPoints[0][1] - 5
    #     chinese_width = picPoints[0][2] + 10
    #     chinese_height = picPoints[0][3] + 10
    #
    #     if (chinese_x_end - chinese_width ) < 0:
    #         chinese_start = 0
    #     else:
    #         chinese_start = chinese_x_end - chinese_width
    #
    # x_points_start = []
    # x_points_end = []
    # y_points_start = []
    # y_points_end = []
    # testPic = cv2.rectangle(testPic, (chinese_start, chinese_y_point), (chinese_x_end , chinese_y_point + chinese_height ), (0, 255, 0), 2)
    #
    # x_points_start.append(chinese_start)
    # x_points_end.append(chinese_x_end)
    # y_points_start.append(chinese_y_point)
    # y_points_end.append(chinese_y_point + chinese_height)
    #
    # # x_points_start = []
    # # x_points_end = []
    # # y_start = picPoints[0][1] - 5
    # # y_end = picPoints[0][1] + picPoints[0][3] + 5
    #
    # for xx, yy, ww, hh in picPoints:
    #     if xx - 5 < 0:
    #         start = 0
    #     else:
    #         start = xx - 5
    #
    #     testPic = cv2.rectangle(testPic, (start, yy - 5), (xx + ww + 5, yy + hh + 5), (0, 255, 0), 2)
    #
    #     x_points_start.append(start)
    #     x_points_end.append(xx + ww + 5)
    #     y_points_start.append(yy - 5)
    #     y_points_end.append(yy + hh + 5)
    #
    # print("start: ", x_points_start, "ENd: ", x_points_end)
    # print('------------------------------')
    # print("start: ", y_points_start, "ENd: ", y_points_end)
        # x_points_start.append(xx - 5)
        # x_points_end.append(xx + ww + 5)
    # print(x_points_start)
    # print(x_points_end)

    # firstWidth = x_points_end[0] - x_points_start[0]
    #
    # firstX = x_points_start[0] - firstWidth
    # firstX_End = x_points_start[0]
    # print(firstX, firstX_End)

    # chinese_pic = None

    # 判断汉字左边和上边越界问题
    # if firstX < 0:
    #     if y_start < 0:
    #         cv2.imshow("FirstNumber", binary[0:y_end, 0:firstX_End])
    #         chinese_pic = binary[0:y_end, 0:firstX_End].copy()
    #     else:
    #         cv2.imshow("FirstNumber", binary[y_start:y_end, 0:firstX_End])
    #         chinese_pic = binary[y_start:y_end, 0:firstX_End].copy()
    # else:
    #     if y_start < 0:
    #         cv2.imshow("FirstNumber", binary[0:y_end, firstX:firstX_End])
    #         chinese_pic = binary[0:y_end, firstX:firstX_End].copy()
    #     else:
    #         cv2.imshow("FirstNumber", binary[y_start:y_end, firstX:firstX_End])
    #         chinese_pic = binary[y_start:y_end, firstX:firstX_End].copy()

    # cv2.imshow('test', testPic)
    # # cv2.imshow('chinese', chinese_pic)
    # # cv2.imwrite('p0.jpg', chinese_pic)
    #
    # #
    # for i in range(7):
    #     cv2.imshow("第" + str(i) + "个", binary[y_points_start[i]:y_points_end[i], x_points_start[i]:x_points_end[i]])
    #     # cv2.imwrite('p'+str(i+1)+'.jpg', binary[0:y_end, x_points_start[i]:x_points_end[i]])
    #
    # cv2.waitKey(0)

if __name__ == '__main__':
    img = cv2.imread("6.jpg")
    angle, angle_flag = detect(img)
    cutNoAnglePlate(angle, angle_flag, img)