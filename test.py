import numpy as np
import cv2
from PIL import Image
import os
import kagglehub

if __name__ == '__main__':
    for i in range(3):
        with open(f'{i}.txt', 'w') as f:
            f.write('0 ')

    '''
    # find countours
    # binary picture
    image = cv2.imread('1.png')
    height, width = image.shape[:2]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('binary_image.jpg', binary_image)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 打开txt文件准备写入
    with open('poly.txt', 'w') as f:
        f.write('0 ')
        for contour in contours:
            # 轮廓近似的精度参数
            epsilon = 0.005 * cv2.arcLength(contour, True)  # 注意：这里使用contour而不是contours[0]
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            # 将轮廓点转换为顶点坐标，并去除重复点
            approx_contour = approx_contour.reshape(-1, 2)
            approx_contour = np.unique(approx_contour, axis=0)
            print('contour.shape=', approx_contour.shape)
            # 将顶点坐标写入txt文件
            for point in approx_contour:
                x, y = point
                x = x/width
                y = y/height
                f.write(f'{x} {y} ')

    # 将轮廓画回图像
    for contour in contours:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        contour_image = cv2.drawContours(image.copy(), [approx_contour], -1, (0, 255, 0), 2)  # 绿色轮廓线，线宽为2

    # 保存带有轮廓的图像
    cv2.imwrite('contours_image.jpg', contour_image)

    # 显示带有轮廓的图像
    cv2.imshow('Contours Image', contour_image)
    cv2.imshow('Binary Image', binary_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''