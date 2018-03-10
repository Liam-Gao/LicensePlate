import cv2
import numpy as np
import operator
from sklearn.linear_model import LogisticRegression


def whitePointFeature(img):
    row, col = img.shape
    feature = []
    for i in range(row):
        count = 0
        for j in range(col):
            if img[i][j] > 0:
                count += 1
        feature.append(count)

    for i in range(col):
        count = 0
        for j in range(row):
            if img[j][i] > 0:
                count += 1
        feature.append(count)
    return feature


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 中值滤波或许可以在车牌部分使用
    equ = cv2.equalizeHist(gray)
    gaussian = cv2.GaussianBlur(equ, (5, 5), 0, 0, cv2.BORDER_DEFAULT)

    # sobel或许可以在车牌部分使用
    sobel = cv2.Sobel(gaussian, cv2.CV_8U, 1, 0, ksize=3)
    ret, binary = cv2.threshold(sobel, 125, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))

    dilation = cv2.dilate(binary, element1, iterations=1)
    erosion = cv2.erode(dilation, element1, iterations=1)
    dilation2 = cv2.dilate(erosion, element2, iterations=3)

    cv2.imshow('e', dilation2)
    cv2.waitKey()
    return dilation2


def hsvprocess(img, isHsvOr_S):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 为True返回S通道，为False返回HSV
    if isHsvOr_S:
        h, s, v = cv2.split(hsv)
        ret, binary = cv2.threshold(s, 100, 255, cv2.THRESH_BINARY)
        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
        element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        dilation = cv2.dilate(binary, element1, iterations=1)
        erosion = cv2.erode(dilation, element2, iterations=2)
        # element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        # dilation = cv2.dilate(binary, element1, iterations=1)
        cv2.imshow('S_HSV', erosion)
        cv2.waitKey()
        return dilation
    else:
        lower_blue = np.array([80, 43, 46])
        upper_blue = np.array([115, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        dilation = cv2.dilate(mask, element1, iterations=1)
        cv2.imshow('Blue_HSV', dilation)
        cv2.waitKey()
        return dilation


def findPlateNumberRegion(combImg):
    region = []
    returnimg, contours, hierarchy = cv2.findContours(combImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if (area < 2000):
            continue

        # 轮廓近似，作用很小
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(approx)
        angle = rect[2]
        print('rect angle is:', angle)
        if angle > -80 and angle < -30:
            continue
        box = cv2.boxPoints(rect)

        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        ratio = float(width) / float(height)
        if (ratio > 5.0 or ratio < 1.3):
            continue
        region.append(box)
    return region


def writeRegions(region, img):
    newImg = img.copy()
    for box in region:
        ctr = np.array(box).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(newImg, [ctr], 0, (0, 255, 0), 2)

        ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
        xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
        # argsort函数返回的是数组值从小到大的索引值
        ys_sorted_index = np.argsort(ys)
        xs_sorted_index = np.argsort(xs)
        # print('--------',xs_sorted_index, ys_sorted_index)
        x1 = box[xs_sorted_index[0], 0]
        x2 = box[xs_sorted_index[3], 0]

        y1 = box[ys_sorted_index[0], 1]
        y2 = box[ys_sorted_index[3], 1]

        img_plate = img[int(y1):int(y2), int(x1):int(x2)]
        cv2.imshow('plate', newImg)
        cv2.imshow('number plate', img_plate)
        cv2.imwrite('thePlate.jpg', img_plate)

        rect = cv2.minAreaRect(box)
        angle = rect[2]
    cv2.waitKey()
    return angle


def cutBlack(binary):
    row, col = binary.shape
    for i in range(row):
        count = 0
        for j in range(col):
            if binary[i][j] != 0:
                count += 1
        if count > col * 0.1:
            upLine = i
            break
    for i2 in range(row - 1, 0, -1):
        count = 0
        for j2 in range(col):
            if binary[i2][j2] != 0:
                count += 1
        if count > col * 0.1:
            downLine = i2
            break
    for i3 in range(col):
        count = 0
        for j3 in range(row):
            if binary[j3][i3] != 0:
                count += 1
        if count > col * 0.1:
            leftLine = i3
            break
    for i4 in range(col - 1, 0, -1):
        count = 0
        for j4 in range(row):
            if binary[j4][i4] != 0:
                count += 1
        if count > col * 0.1:
            rightLine = i4
            break
    return upLine, downLine, leftLine, rightLine


def findBlackLength(row, col, binary):
    count = 0
    for i in range(col):
        # print(i)
        if binary[row][i] != 0:
            count += 1
        else:
            break
    return count


def affineFunction(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    gaussian = cv2.GaussianBlur(equ, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    ret, binary = cv2.threshold(gaussian, 125, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    cv2.imshow('cubinary', binary)

    row, col = binary.shape
    row1 = int(row * 0.25)
    row2 = int(row * 0.5)
    row3 = int(row * 0.75)
    row1Length = findBlackLength(row1, col, binary)
    row2Length = findBlackLength(row2, col, binary)
    row3Length = findBlackLength(row3, col, binary)

    # 判断是斜率是正True还是负False，小于一定度数就不进行仿射变换
    angle_flag = False
    if row1Length > row3Length:
        angle_flag = True

    slope1 = abs(row1Length - row3Length) / abs(row3 - row1)
    slope2 = abs(row1Length - row2Length) / abs(row2 - row1)
    slope3 = abs(row2Length - row3Length) / abs(row3 - row2)
    slope = (slope1 + slope2 + slope3) / 3

    # 斜率大于0.1就进行仿射变换
    if slope > 0.1:
        x_move = int(row * slope)
        if angle_flag:
            pts1 = np.float32([[0, row - 1], [x_move, 0], [col - 1, 0]])
            pts2 = np.float32([[0, row - 1], [0, 0], [col - 1 - x_move, 0]])
        else:
            pts1 = np.float32([[0, 0], [x_move, row - 1], [col - 1, row - 1]])
            pts2 = np.float32([[0, 0], [0, row - 1], [col - 1 - x_move, row - 1]])

        M = cv2.getAffineTransform(pts1, pts2)
        affine = cv2.warpAffine(img, M, (col, row))
        cv2.imshow('affine', affine)
    else:
        affine = img
    cv2.waitKey()
    return affine


def cutLeftRightBlack(binary):
    row, col = binary.shape
    for i3 in range(col):
        count = 0
        for j3 in range(row):
            if binary[j3][i3] != 0:
                count += 1
        if count > col * 0.1:
            leftLine = i3
            break
    for i4 in range(col - 1, 0, -1):
        count = 0
        for j4 in range(row):
            if binary[j4][i4] != 0:
                count += 1
        if count > col * 0.1:
            rightLine = i4
            break
    return leftLine, rightLine


def cutPlateFunction(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # equ = cv2.equalizeHist(gray)
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    ret, binary = cv2.threshold(gaussian, 150, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    cv2.imshow('cutPlate', binary)
    cv2.waitKey()

    # 去除杂质走一波
    # 先去处黑边
    hsv = hsvprocess(img, False)
    left, right = cutLeftRightBlack(hsv)
    newImg = img[:, left:right]
    newBinary = binary[:, left:right]
    cv2.imshow('newImg', newBinary)
    cv2.waitKey()
    #  去除较少的白点的行
    nrow, ncol = newBinary.shape
    for i in range(nrow):
        count = 0
        for j in range(ncol):
            if newBinary[i][j] == 255:
                count += 1
        if count < ncol * 0.1:
            for j in range(ncol):
                newBinary[i][j] = 0
    cv2.imshow('newCutImg', newBinary)
    cv2.waitKey()

    # 开始画轮廓，取字符
    reImg, contours, hierarchy = cv2.findContours(newBinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    picPoints = []
    for contour in contours:
        if cv2.contourArea(contour) < 600:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if h < nrow / 2.5:
            continue
        if w < 10 or w > ncol * 1 / 4:
            continue
        # print(cv2.contourArea(contour))
        # print(x, y, w, h)
        picPoints.append([x, y, w, h])

    picPoints.sort(key=operator.itemgetter(0))
    print(picPoints)

    # 去掉汉字的框选
    if len(picPoints) == 7:
        picPoints.remove(picPoints[0])
    print(picPoints)
    # 字符之间的间隔
    gap = picPoints[2][0] - picPoints[1][0] - picPoints[1][2]
    print("gap is ", gap)

    chineseX = picPoints[0][0] - gap - picPoints[0][2]
    chineseY = picPoints[0][1]
    chineseW = picPoints[0][2]
    chineseH = picPoints[0][3]
    drawImg = cv2.rectangle(newImg, (chineseX, chineseY), (chineseX + chineseW, chineseY + chineseH), (0, 255, 0), 1)
    wLenthMax = picPoints.copy()
    wLenthMax.sort(key=operator.itemgetter(2), reverse=True)
    print(wLenthMax)
    for xx, yy, ww, hh in picPoints:
        if ww < 15:
            ww = wLenthMax[0][2]
            xx = xx - ww // 2 + 3
        drawImg = cv2.rectangle(newImg, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 1)
    cv2.imshow('testPic', drawImg)
    cv2.waitKey()

    # 开始分割字符并write出来
    cv2.imwrite('p0.jpg', newBinary[chineseY:chineseY + chineseH, chineseX:chineseX + chineseW])
    for i in range(6):
        if picPoints[i][2] < 15:
            picPoints[i][2] = wLenthMax[0][2]
            picPoints[i][0] = picPoints[i][0] - wLenthMax[0][2] // 2 + 3
        cv2.imwrite('p' + str(i + 1) + '.jpg', newBinary[picPoints[i][1]:picPoints[i][1] + picPoints[i][3],
                                               picPoints[i][0]:picPoints[i][0] + picPoints[i][2]])


def recongnitionPlate():
    r = np.load('trainData.npz')
    feature = r["arr_0"]
    labels = r["arr_1"]

    lg = LogisticRegression()
    lg.fit(feature, labels)

    c_dic = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
             '11': 'A', '12': 'B', '13': 'C', '14': 'D', '15': 'E', '16': 'F', '17': 'G', '18': 'H', '19': 'J',
             '20': 'K', '21': 'L', '22': 'M', '23': 'N', '24': 'P', '25': 'Q', '26': 'R', '27': 'S', '28': 'T',
             '29': 'U', '30': 'V', '31': 'W', '32': 'X', '33': 'Y', '34': 'Z'}

    for i in range(6):
        img = cv2.imread('p' + str(i + 1) + '.jpg', 0)
        img = cv2.resize(img, (47, 92))
        ret, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        pre = np.array(whitePointFeature(binary)).reshape(1, -1)
        # # print(feature.shape, labels.shape, pre.shape)
        print(c_dic[lg.predict(pre)[0]])


if __name__ == '__main__':
    img = cv2.imread('license4.png')
    # 形态学处理
    dilation = preprocess(img)
    hsv = hsvprocess(img, False)
    combImg = cv2.bitwise_and(dilation, hsv)
    cv2.imshow('comb', combImg)
    cv2.waitKey()

    region = findPlateNumberRegion(combImg)
    angle = writeRegions(region, img)

    # 车牌进行旋转纠正 , 还存在未验证的部分角度
    plate = cv2.imread('thePlate.jpg')
    row, col = plate.shape[:2]
    if angle < -6:
        if angle > -50:
            M = cv2.getRotationMatrix2D((col // 2, row // 2), angle, 1)
        else:
            angle = -(90 + angle)
            M = cv2.getRotationMatrix2D((col // 2, row // 2), -angle, 1)

        dst = cv2.warpAffine(plate, M, (col, row))
        cv2.imshow('aaa', dst)
        cv2.waitKey()
    else:
        dst = plate
    # -----------------------------------------------------------------------------------------------------
    # 再次进行颜色定位，消除车牌部分干扰
    p_hsv = hsvprocess(dst, True)
    # 进行切割黑色部分
    row1, row2, col1, col2 = cutBlack(p_hsv)
    newDist = dst[row1:row2, col1:col2]
    cv2.imshow('ndist', newDist)
    cv2.waitKey()

    # 通过仿射变换进行倾斜纠正
    affinePlate = affineFunction(newDist)

    bigPlate = cv2.resize(affinePlate, (440, 140))
    # 去除车牌杂质，勾画轮廓，并分割车牌
    cutPlateFunction(bigPlate)
    # 进行识别
    recongnitionPlate()
