import cv2
import numpy as np


def preprocess(gray):
    # 直方图均衡化
    equ = cv2.equalizeHist(gray)
    # 高斯平滑
    # 高斯模糊本质上是低通滤波器，输出图像的每个像素点是原图像上对应像素点与周围像素点的加权和
    # (3,3)表示矩阵长宽,后面两个分别是X、Y方向的高斯核标准差
    gaussian = cv2.GaussianBlur(equ, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    # 中值滤波
    median = cv2.medianBlur(gaussian, 5)
    # Sobel算子，X方向求梯度
    sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize=3)
    # 二值化
    ret, binary = cv2.threshold(sobel, 125, 255, cv2.THRESH_BINARY)
    # 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)
    # 腐蚀一次，去掉细节
    erosion = cv2.erode(dilation, element1, iterations=1)
    # 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=3)
    # cv2.imshow('binary', binary)
    # cv2.imshow('dilation', dilation2)
    # cv2.imshow('median2', median)
    # cv2.imshow('sobel3', sobel)
    cv2.imshow('dilation2', dilation2)
    # cv2.imshow('erosion', erosion)
    cv2.waitKey(0)
    return dilation2


def BlueHsvImg(img):
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #
    # lower_blue = np.array([100, 43, 46])
    # upper_blue = np.array([130, 255, 255])
    # mask = cv2.inRange(hsv, lower_blue, upper_blue)
    #
    # element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # dilation = cv2.dilate(mask, element1, iterations=1)

    # return dilation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    ret, binary = cv2.threshold(s, 100, 255, cv2.THRESH_BINARY)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    dilation = cv2.dilate(binary, element2, iterations=1)
    erosion = cv2.erode(dilation, element1, iterations=1)
    cv2.imshow('errr', erosion)
    return erosion

def findPlateNumberRegion(img):
    region = []
    # 查找轮廓
    """
    第一个，也是最坑爹的一个，它返回了你所处理的图像
    第二个，正是我们要找的，轮廓的点集
    第三个，各层轮廓的索引
    """
    returnimg, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow('asdf', contours)
    # cv2.waitKey(0)
    # 筛选面积小的
    # print("countours: ", contours)
    for i in range(len(contours)):
        cnt = contours[i]
        # print('cnt: ', cnt)
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)
        # print('areas: ', area)
        # 面积小的都筛选掉
        if (area < 2000):
            continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(approx)
        # print("rect is: ", rect)

        # box是四个点的坐标
        """
        minAreaRect返回的是一个 Box2D 结构,其中包含
        矩形左上角角点的坐标(x,y),矩形的宽和高(w,h),以及旋转角度。但是
        要绘制这个矩形需要矩形的 4 个角点,可以通过函数 cv2.boxPoints() 获得。
        """

        # box = cv2.cv.BoxPoints(rect)
        box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # print('box: ', box)
        # print('boxx: ', box[0][1])

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        # 车牌正常情况下长高比在2.7-5之间
        ratio = float(width) / float(height)
        # print('ratio: ', ratio)
        if (ratio > 5.0 or ratio < 1.4):
            continue
        region.append(box)
    # print('region: ', region.__len__())
    return region


def detect(img):
    newImg = img.copy()
    # 转化成灰度图
    gray = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)

    # 形态学变换的预处理
    dilation = preprocess(gray)

    # HSV处理
    hsvBinary = BlueHsvImg(newImg)

    cv2.imshow('aa', hsvBinary)
    getCombImg = cv2.bitwise_and(dilation, hsvBinary)
    cv2.imshow('combbb', getCombImg)
    cv2.waitKey(0)

    # 查找车牌区域
    region = findPlateNumberRegion(getCombImg)

    # 用绿线画出这些找到的轮廓
    print('this is region', region)
    for box in region:
        """
        you should convert your contour to numpy array first
        contours is a Python list of all the contours in the image. Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
        """
        # print('box', box)
        ctr = np.array(box).reshape((-1, 1, 2)).astype(np.int32)
        # print("asdas", [ctr])
        cv2.drawContours(newImg, [ctr], 0, (0, 255, 0), 2)
        rectt = cv2.minAreaRect(box)
        angle = rectt[2]
        # 矩形倾斜方向判断，往左上为True
        flag = box[0][1] - box[3][1]
        if flag >= 0:
            angle_flag = True
        else:
            angle_flag = False
        # cv2.boxPoints(rectt)
        # print('----this is minrect: ', cv2.minAreaRect(box))
    cv2.imshow('testt', newImg)
    ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
    xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
    # argsort函数返回的是数组值从小到大的索引值
    ys_sorted_index = np.argsort(ys)
    xs_sorted_index = np.argsort(xs)
    print('--------',xs_sorted_index, ys_sorted_index)
    x1 = box[xs_sorted_index[0], 0]
    x2 = box[xs_sorted_index[3], 0]

    y1 = box[ys_sorted_index[0], 1]
    y2 = box[ys_sorted_index[3], 1]
    print('0----0: ', x1,x2,y1,y2)
    # img_org2 = img.copy()
    img_plate = img[int(y1):int(y2), int(x1):int(x2)]
    cv2.imshow('number plate', img_plate)
    cv2.imwrite('number_plate1.jpg', img_plate)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)

    # 带轮廓的图片
    cv2.imwrite('contours.png', newImg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return angle, angle_flag


if __name__ == '__main__':
    # imgg = cv2.imread("license3.jpeg")
    imgg = cv2.imread("license4.png")
    # imgg = cv2.imread("license.jpeg")
    # imgg = cv2.imread("timg.jpg")
    # imgg = cv2.imread("cccc.jpeg")
    # imgg = cv2.imread("cuan.jpeg")

    angle = detect(imgg)