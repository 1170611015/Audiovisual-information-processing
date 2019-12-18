import cv2
import numpy as np
import math

def robert_algo(image):
    B,G,R = cv2.split(image)
    first_array = [[1,0],[0,-1]]
    second_array = [[0,-1],[1,0]]
    #检测蓝色通道边缘
    h = B.shape[0]
    w = B.shape[1]
    new_B = B
    for i in range(h-1):
        for j in range(w-1):
            temp_array = B[i:i+2,j:j+2]
            value = np.abs(np.sum(temp_array * first_array)) + np.abs(np.sum(temp_array * second_array))
            if value > 255:
                value = 255
            new_B[i][j] = value
    # 检测绿色通道边缘
    h = G.shape[0]
    w = G.shape[1]
    new_G = G
    for i in range(h - 1):
        for j in range(w - 1):
            temp_array = G[i:i + 2, j:j + 2]
            value = np.abs(np.sum(temp_array * first_array)) + np.abs(np.sum(temp_array * second_array))
            if value > 255:
                value = 255
            new_G[i][j] = value
    # 检测红色通道边缘
    h = R.shape[0]
    w = R.shape[1]
    new_R = R
    for i in range(h - 1):
        for j in range(w - 1):
            temp_array = R[i:i + 2, j:j + 2]
            value = np.abs(np.sum(temp_array * first_array)) + np.abs(np.sum(temp_array * second_array))
            if value > 255:
                value = 255
            new_R[i][j] = value
    return cv2.merge([new_B,new_G,new_R])


def gray_sobel_sharp(gray_img, throd, norm=1):
    """"
    对于单通道的图像(灰度图)使用Sobel算子进行锐化
    """
    h, w = np.shape(gray_img)
    img_sharp = np.zeros((h - 2, w - 2))

    # 水平和垂直的Sobel算子
    r_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    c_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            array = [np.sum(gray_img[i - 1:i + 2, j - 1:j + 2] * r_filter),
                     np.sum(gray_img[i - 1:i + 2, j - 1:j + 2] * c_filter)]
            img_sharp[i - 1, j - 1] = np.linalg.norm(array, ord=norm)
    img_sharp = np.where(img_sharp > throd, img_sharp, 0)
    return img_sharp

def rbg_sober_sharp(image):
    B,G,R = cv2.split(image)
    new_B = gray_sobel_sharp(B,120,1)
    new_G = gray_sobel_sharp(G,120,1)
    new_R = gray_sobel_sharp(R,120,1)
    return cv2.merge([new_B,new_G,new_R])
