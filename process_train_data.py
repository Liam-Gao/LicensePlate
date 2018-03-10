import cv2
import numpy as np
import os



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

# -------------------------------除了汉字外的所有数字字母集合----------------------------------
# for root, dirs, files in os.walk('/home/liamgao/mydata/temp'):
#     feature = []
#     labels = []
#     for i in range(len(files)):
#         a = files[i]
#         img = cv2.imread('/home/liamgao/mydata/temp' + '/' + a, 0)
#         ret, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
#         feature.append(whitePointFeature(binary))
#         labels.append(a.split('.')[0].split('_')[0])
# #
#     feature = np.array(feature)
#     labels = np.array(labels)
#     print(len(feature))
#     print(len(labels))
#     np.savez('trainData.npz', feature, labels)

# img = cv2.imread('1_50.jpg')
# cv2.imwrite('1_50_1.jpg', cv2.resize(img, (47, 92)))

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
