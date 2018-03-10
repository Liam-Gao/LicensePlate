import cv2
import math
import operator
import numpy as np

# def processPic(img):
#     # 再次裁剪图片
#     # 裁剪上下方的空白
#     row, col = img.shape
#     start = end = 0
#     for i in range(row):
#         count = 0
#         for j in range(col):
#             if img[i][j] != 0:
#                 count += 1
#         if count != 0:
#             start = i
#             break
#
#     for i in range(row - 1, 0, -1):
#         count = 0
#         for j in range(col):
#             if img[i][j] != 0:
#                 count += 1
#         if count != 0:
#             end = i
#             break
#     return img[start:end, :]



def countWhitePoints(img):
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


def eulideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def calculateDistance(chinese_list, iden_list):
    result_list = []
    for i in range(len(chinese_list)):
        distance = 0
        result = []
        for j in range(72):
            distance += pow((int(chinese_list[i][j]) - iden_list[j]), 2)
        result.append(math.sqrt(distance))
        result.append(chinese_list[i][-1])
        result_list.append(result)
    # print(result_list)
    return result_list


def getPrediction(result, k):
    neighbors = []
    classVotes = {}
    for i in range(k):
        neighbors.append(result[i])
    for j in range(len(neighbors)):
        response = str(int(neighbors[j][-1]))
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


# ------------------------------------加载特征--------------------------------------
# chinese_list = []
# f = loadFeatures('feature_chinese.txt')
# for line in f:
#     p = line.split(',')
#     chinese_list.append(p)
#
# letter_list = []
# f = loadFeatures('feature_letter.txt')
# for line in f:
#     p = line.split(',')
#     letter_list.append(p)
#
# num_letter_list = []
# f = loadFeatures('feature_num_letter.txt')
# for line in f:
#     p = line.split(',')
#     num_letter_list.append(p)
# -----------------------------------------------------------------------------------

r = np.load('trainData.npz')
feature = r["arr_0"]
labels = r["arr_1"]
img = cv2.imread('p5.jpg', 0)
img = cv2.resize(img, (92, 47))
ret, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
iden_list = countWhitePoints(binary)
result = calculateDistance(feature, iden_list)
# img = cv2.imread('p1.jpg', 0)
# for i in range(7):
#     img = cv2.imread('p' + str(i) + '.jpg', 0)
#     ret, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
#     p_binary = processPic(binary)
#
#     re_binary = cv2.resize(p_binary, (32, 40))
#     cv2.imshow('a', re_binary)
#     cv2.waitKey()
#     iden_list = countWhitePoints(re_binary)
#     if i == 0:
#         result = calculateDistance(chinese_list, iden_list)
#     elif i == 1:
#         result = calculateDistance(letter_list, iden_list)
#     else:
#         result = calculateDistance(num_letter_list, iden_list)
#
#     result.sort(key=operator.itemgetter(0))
#     k = 5
#     prediction = str(int(getPrediction(result, k)))
    # print('----',prediction)
    # ----------------------------------------DIC------------------------------------
    # if i == 0:
    #     c_dic = {'0': '京', '1': '闽', '2': '粤', '3': '苏', '4': '沪', '5': '浙'}
    #
    # elif i == 1:
    #     c_dic = {'10': 'A', '11': 'B', '12': 'C', '13': 'D', '14': 'E', '15': 'F', '16': 'G', '17': 'H', '18': 'J',
    #              '19': 'K', '20': 'L', '21': 'M', '22': 'N', '23': 'P', '24': 'Q', '25': 'R', '26': 'S', '27': 'T',
    #              '28': 'U', '29': 'V', '30': 'W', '31': 'X', '32': 'Y', '33': 'Z'}
    # else:
    #     c_dic = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
    #              '10': 'A', '11': 'B', '12': 'C', '13': 'D', '14': 'E', '15': 'F', '16': 'G', '17': 'H', '18': 'J',
    #              '19': 'K', '20': 'L', '21': 'M', '22': 'N', '23': 'P', '24': 'Q', '25': 'R', '26': 'S', '27': 'T',
    #              '28': 'U', '29': 'V', '30': 'W', '31': 'X', '32': 'Y', '33': 'Z'}
    # print(c_dic[prediction])
cv2.waitKey()
