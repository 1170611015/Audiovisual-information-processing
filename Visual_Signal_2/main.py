import cv2
import numpy
from Filter import median_filter
from Filter import mean_filter
from add_noise import add_salt_noise
from add_noise import add_gaussian_noise
from edge_detection import robert_algo
from edge_detection import  gray_sobel_sharp
from edge_detection import  rbg_sober_sharp

srca = cv2.imread('C:\\Users\\liangmingliang\\PycharmProjects\\Visual_signal_!\\photo.png')





image_gray = cv2.cvtColor(srca,cv2.COLOR_BGR2GRAY)
new_gray_image = gray_sobel_sharp(image_gray,120)
sobel_image = rbg_sober_sharp(srca)
cv2.imwrite('sobel_image.bmp',sobel_image)
cv2.waitKey(0)
cv2.destroyAllWindows()