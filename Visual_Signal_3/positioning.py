import numpy as np
import cv2
import math
import os

minPlateRatio = 0.5  # 车牌最小比例
maxPlateRatio = 5  # 车牌最大比例


# 找到符合车牌形状的矩形
def findPlateNumberRegion(img):
    region = []
    #获得车牌轮廓的集合
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    list_rate = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        #如果面积小于300，过于小，说明不是车牌
        if area < 200:
            continue
        rect = cv2.minAreaRect(cnt)
        box = np.int32(cv2.boxPoints(rect))
        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        ratio = float(width) / float(height)
        rate = getxyRate(cnt)
        #长宽比例要符合要求
        if ratio > maxPlateRatio or ratio < minPlateRatio:
            continue
        region.append(box)
        list_rate.append(ratio)
    index = getSatifyestBox(list_rate)
    return region[index]

#选取最合适的车牌
def getSatifyestBox(list_rate):
    for index, key in enumerate(list_rate):
        list_rate[index] = abs(key - 3)
    index = list_rate.index(min(list_rate))
    return index

#获得车牌的长宽比例
def getxyRate(cnt):
    x_list = []
    y_list = []
    for location_value in cnt:
        location = location_value[0]
        x_list.append(location[0])
        y_list.append(location[1])
    x_height = max(x_list) - min(x_list)
    y_height = max(y_list) - min(y_list)
    return x_height * (1.0) / y_height * (1.0)

#中值滤波
def MeanSmooth(gray_image):
    blur_image = gray_image
    gaussian_matrix = [[2,2,2],[2,2,2],[2,2,2]]
    for i in range(1,gray_image.shape[0]-1):
        for j in range(1,gray_image.shape[1]-1):
            blur_image[i][j] = (np.sum(gray_image[i-1:i+2,j-1:j+2] * gaussian_matrix))/18
    return blur_image

#sobel算子，提取边缘
def SobelAlgorithm(grayimage):
    sobel_matrix = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sobel_image = grayimage
    for i in range(1,grayimage.shape[0]-1):
        for j in range(1,grayimage.shape[1]-1):
            sobel_image[i-1][j-1] = np.abs(np.sum(grayimage[i-1:i+2,j-1:j+2] * sobel_matrix))
            if sobel_image[i-1][j-1] < 100:
                sobel_image[i-1][j-1] = 0
            else:
                sobel_image[i-1][j-1] = sobel_image[i-1][j-1]
    return  sobel_image

#腐蚀
def erosion(binaryimage,struct_element):
    erosion_image = np.zeros((binaryimage.shape[0],binaryimage.shape[1]),np.uint8)
    s = np.int((struct_element[0] - 1) / 2)
    t = np.int((struct_element[1] - 1) / 2)
    for i in range(s,binaryimage.shape[0]-s):
        for j in range(t,binaryimage.shape[1]-t):
            flag = 0
            for k in range(i-s,i+s+1):
                for l in range(j-t,j+t+1):
                    if binaryimage[k][l] == 0:
                        flag = 1
                    else:
                        flag = 0
            if flag == 1:
                erosion_image[i][j] = 0
            else:
                erosion_image[i][j] = 255
    print("erosion",erosion_image)
    cv2.imshow("erosion",erosion_image)
    cv2.waitKey(0)
    return erosion_image

#扩充
def dilation(binaryimage,struct_element):
    dilation_image = np.zeros((binaryimage.shape[0],binaryimage.shape[1]),np.uint8)
    s = np.int((struct_element[0] - 1) / 2)
    t = np.int((struct_element[1] - 1) / 2)
    for i in range(s, binaryimage.shape[0] - s ):
        for j in range(t, binaryimage.shape[1] - t ):
            flag = 0
            for k in range(i - s, i + s + 1):
                for l in range(j - t, j + t + 1):
                    if binaryimage[k][l] == 255:
                        flag = 1
            if flag == 1:
                dilation_image[i][j] = 255
            else:
                dilation_image[i][j] = 0
    print("dilation",dilation_image)
    cv2.imshow("dilation",dilation_image)
    cv2.waitKey(0)
    return dilation_image

def ClosedAlgorithm(binaryimage,struct_element):
    binaryimage = dilation(binaryimage,struct_element)
    binaryimage = erosion(binaryimage,struct_element)
    return binaryimage

def OpeningAlgorithm(binaryimage,struct_element):
    binaryimage = erosion(binaryimage,struct_element)
    binaryimage = dilation(binaryimage,struct_element)
    return binaryimage






def find_license(file):
    raw_image = cv2.imread(file)
    # 灰度化
    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    cv2.waitKey(0)
    # 中值滤波，车牌识别中利用中值滤波将图片平滑化，去除干扰的噪声对后续图像处理的影响
    smooth_image = MeanSmooth(gray)
    cv2.imshow("smooth_image", smooth_image)
    cv2.waitKey(0)
    #sobel算子：车牌定位的核心算法，水平方向上的边缘检测，检测出车牌区域
    sobel = SobelAlgorithm(smooth_image)
    cv2.imshow("sobel", sobel)
    cv2.waitKey(0)
    #二值化
    ret, binary = cv2.threshold(sobel, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("binary", binary)
    cv2.waitKey(0)
    #获得结构元素
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    #element = (9,3)
    #闭操作
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)
    #closed = ClosedAlgorithm(binary,element)
    print(closed)
    cv2.imshow("closed", closed)
    cv2.waitKey(0)
    region = findPlateNumberRegion(closed)
    cv2.drawContours(raw_image, [region], 0, (0, 255, 0), 2)
    cv2.imwrite("C:\\Users\\liangmingliang\\PycharmProjects\\Visual_Signal_Third\\5_result.bmp",raw_image)
    cv2.imshow("img",raw_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    file = "C:\\Users\\liangmingliang\\PycharmProjects\\Visual_Signal_Third\\5.png"
    find_license(file)