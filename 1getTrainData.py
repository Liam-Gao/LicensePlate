import numpy as np
import cv2


myData = []
for line in open('/home/liamgao/车牌模板/sample/chinese/chinese.txt'):
    myData.append(line.split())
print(myData)

picPath = '/home/liamgao/车牌模板/sample/chinese/' + myData[0][0]
print(picPath)

img = cv2.imread(picPath, 0)
imgNumpy = np.array(img)
print(imgNumpy.shape)
testdata = []

# 行白点
for x in range(imgNumpy.shape[0]):
    whiteNumRow = 0
    for y in range(imgNumpy.shape[1]):
        if int(imgNumpy[x][y]) > 175:
            whiteNumRow += 1
    testdata.append(whiteNumRow)
    #     列白点
for x in range(imgNumpy.shape[1]):
    whiteNumCol = 0
    for y in range(imgNumpy.shape[0]):
        if int(imgNumpy[y][x]) > 175:
            whiteNumCol += 1
    testdata.append(whiteNumCol)

print(testdata)

cv2.waitKey()