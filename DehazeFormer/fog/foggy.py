import cv2
import math
import numpy as np
import random

def demo(img_path):
    img = cv2.imread(img_path)
    img_f = img / 255.0 # 归一化
    (row, col, chs) = img.shape

    A = 0.5  # 亮度
    beta = 0.05  # 雾的浓度
    # beta = betaa[random.randint(0,len(betaa)-1)] # 随机初始化雾的浓度
    size = math.sqrt(max(row, col))  # 雾化尺寸，可根据自己的条件进行调节，一般的范围在中心点位置但不是很大，可自己手动设置参数
    # size = 40 # 这是我自己设置的参数，效果很不错
    center = (row // 2, col // 2)  # 雾化中心 就是图片的中心
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td) # 标准光学模型，图片的RGB三通道进行加雾
    cv2.imwrite('1.png', img_f*255) # 图片生成名字，切记务必要回复图片 *255，否则生成图片错误，可以尝试
    cv2.imshow("src", img)
    cv2.imshow("dst", img_f) # 显示图片

if __name__ == '__main__':
    img_path = "0001_c001_00016450_0.jpg"
    demo(img_path)
