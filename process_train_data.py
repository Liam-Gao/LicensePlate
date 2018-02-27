import os
import cv2


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


def write_feature(file_name, f_list, label):
    file = open(file_name, 'a')
    for i in f_list:
        file.write(str(i)+',')
    file.write(str(label))
    file.write('\n')
    file.close()


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
for i in range(10, 34):
    path = '/home/liamgao/train_data/train_data/letter/'+str(i)
    print(path)
    name_list = os.listdir(path)
    for j in range(len(name_list)):
        img = cv2.imread(path + '/' + name_list[j], 0)
        ret, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        write_feature('feature_letter.txt', whitePointFeature(binary), i)

# with open('feature_num_letter.txt', 'r') as file:
#     for line in file.readlines():
#         p = line.split(',')
