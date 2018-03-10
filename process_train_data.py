import cv2
import numpy as np
import os


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


# def write_feature(file_name, f_list, label):
#     file = open(file_name, 'a')
#     for i in f_list:
#         file.write(str(i)+',')
#     file.write(str(label))
#     file.write('\n')
#     file.close()

# 。。。。。。。。。。。。。。。。。。。。。。。。。。。。汉字部分。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。
# for root, dirs, files in os.walk('/home/liamgao/mydata/chinese'):
#     feature = []
#     labels = []
#     for i in range(len(files)):
#         a = files[i]
#         img = cv2.imread('/home/liamgao/mydata/chinese' + '/' + a, 0)
#         ret, binary = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
#         u, d, l, r = cutBlack(binary)
#         binary = binary[u:d, l:r]
#         # print(binary.shape)
#         binary = cv2.resize(binary, (47, 92))
#         # print(binary.shape)
#         # cv2.imshow('bina', binary)
#         # cv2.waitKey()
#         feature.append(whitePointFeature(binary))
#         labels.append(a.split('.')[0].split('_')[0])
#     #
#     feature = np.array(feature)
#     labels = np.array(labels)
#     print(len(feature))
#     print(len(labels))
#     np.savez('trainChineseData.npz', feature, labels)


# -------------------------------除了汉字外的所有数字字母集合----------------------------------
for root, dirs, files in os.walk('/home/liamgao/mydata/temp'):
    feature = []
    labels = []
    for i in range(len(files)):
        a = files[i]
        img = cv2.imread('/home/liamgao/mydata/temp' + '/' + a, 0)
        ret, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        feature.append(whitePointFeature(binary))
        labels.append(a.split('.')[0].split('_')[0])
#
    feature = np.array(feature)
    labels = np.array(labels)
    print(len(feature))
    print(len(labels))
    np.savez('trainData.npz', feature, labels)

# img = cv2.imread('p1.jpg')
# cv2.imwrite('11_40.jpg', cv2.resize(img, (47, 92)))

# r = np.load('trainData.npz')
# feature = r["arr_0"]
# labels = r["arr_1"]
# # print(feature.shape, labels.shape)
#
# # print(feature)
# # print(labels)
# lg = LogisticRegression()
# lg.fit(feature, labels)
# #
# for i in range(6):
#     img = cv2.imread('p' + str(i + 1) + '.jpg', 0)
#     img = cv2.resize(img, (47, 92))
#     ret, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
#     pre = np.array(whitePointFeature(binary)).reshape(1, -1)
#     # # print(feature.shape, labels.shape, pre.shape)
#     print(lg.predict(pre))



# ---------------------------------------number & letter-----------------------------------
# for i in range(34):
#     path = '/home/liamgao/train_data/train_data/'+str(i)
#     print(path)
#     name_list = os.listdir(path)
#     for j in range(len(name_list)):
#         img = cv2.imread(path + '/' + name_list[j], 0)
#         ret, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
#         write_feature('feature_num_letter.txt', whitePointFeature(binary), i)

# --------------------------------------chinese----------------------------------------------
# for i in range(6):
#     path = '/home/liamgao/train_data/train_data/chinese/'+str(i)
#     print(path)
#     name_list = os.listdir(path)
#     for j in range(len(name_list)):
#         img = cv2.imread(path + '/' + name_list[j], 0)
#         ret, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
#         write_feature('feature_chinese.txt', whitePointFeature(binary), i)

# -------------------------------------letter------------------------------------------------
# for i in range(10, 34):
#     path = '/home/liamgao/train_data/train_data/letter/'+str(i)
#     print(path)
#     name_list = os.listdir(path)
#     for j in range(len(name_list)):
#         img = cv2.imread(path + '/' + name_list[j], 0)
#         ret, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
#         write_feature('feature_letter.txt', whitePointFeature(binary), i)

# with open('feature_num_letter.txt', 'r') as file:
#     for line in file.readlines():
#         p = line.split(',')
