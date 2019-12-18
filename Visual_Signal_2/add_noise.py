import cv2
import numpy as np
import random

def clamp(pv):
    """防止溢出"""
    if pv > 255:
        return 255
    elif pv < 0:
        return 0
    else:
        return pv


def add_gaussian_noise(image):
    """添加高斯噪声"""
    h, w, c = image.shape
    for row in range(0, h):
        for col in range(0, w):
            s = np.random.normal(0, 15, 3)  # 产生随机数，每次产生三个
            b = image[row, col, 0]  # blue
            g = image[row, col, 1]  # green
            r = image[row, col, 2]  # red
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    return image

def add_salt_noise(image,n):
    """添加椒盐噪声"""
    h,w,c = image.shape
    for k in range(n):
        row = random.randint(0,h-1)
        col = random.randint(0,w-1)
        image[row,col,0] = 255
        image[row,col,1] = 255
        image[row,col,2] = 255
    for k in range(n):
        row = random.randint(0,h-1)
        col = random.randint(0,w-1)
        image[row,col,0] = 0
        image[row,col,1] = 0
        image[row,col,2] = 0
    return image


"""
srca = cv2.imread('C:\\Users\\liangmingliang\\PycharmProjects\\Visual_signal_!\\photo.png')

#add_gaussian_noise(srca)
new_image = add_salt_noise(srca,3000)
cv2.imshow("salt and pepper",new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""