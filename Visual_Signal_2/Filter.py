import numpy as np
import cv2

def median_filter(image):
    B,G,R = cv2.split(image)
    #对蓝色通道进行中值滤波
    height = B.shape[0]
    width = B.shape[1]
    for i in range(1,height-1):
        for j in range(1,width-1):
            temparray = np.array([B[i-1][j],B[i][j-1],B[i][j+1],B[i+1][j],B[i][j]])
            B[i][j] = np.uint8(np.median(temparray))
    #对绿色通道进行中值滤波
    height = G.shape[0]
    width = G.shape[1]
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            temparray = np.array([G[i - 1][j], G[i][j - 1], G[i][j + 1], G[i + 1][j], G[i][j]])
            G[i][j] = np.int8(np.median(temparray))

    #对红色通道进行中值滤波
    height = R.shape[0]
    width = R.shape[1]
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            temparray = np.array([R[i - 1][j], R[i][j - 1], R[i][j + 1], R[i + 1][j], R[i][j]])
            R[i][j] = np.uint8(np.median(temparray))
    newimage = cv2.merge([B,G,R])
    return newimage

def mean_filter(image):
    B,G,R = cv2.split(image)
    # 对蓝色通道进行均值滤波
    height = B.shape[0]
    width = B.shape[1]
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            temparray = np.array([B[i - 1][j], B[i][j - 1], B[i][j + 1], B[i + 1][j], B[i][j]])
            B[i][j] = np.uint8(np.mean(temparray))
    # 对绿色通道进行均值滤波
    height = G.shape[0]
    width = G.shape[1]
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            temparray = np.array([G[i - 1][j], G[i][j - 1], G[i][j + 1], G[i + 1][j], G[i][j]])
            G[i][j] = np.uint8(np.mean(temparray))

    # 对红色通道进行均值滤波
    height = R.shape[0]
    width = R.shape[1]
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            temparray = np.array([R[i - 1][j], R[i][j - 1], R[i][j + 1], R[i + 1][j], R[i][j]])
            R[i][j] = np.uint8(np.mean(temparray))
    newimage = cv2.merge([B, G, R])
    return newimage