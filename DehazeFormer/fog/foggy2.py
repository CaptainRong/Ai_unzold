import numpy as np
import cv2
import os
import glob
from tqdm import tqdm  # 导入 tqdm 库

def addfogv2(image, beta, brightness):
    """对输入的图像添加雾霾效果的高效实现。"""
    img_f = image.astype(np.float32) / 255.0
    row, col, chs = image.shape
    size = np.sqrt(max(row, col))
    center = (row // 2, col // 2)
    y, x = np.ogrid[:row, :col]
    dist = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

    # 生成随机噪声
    noise = np.random.normal(loc=0.0, scale=np.random.uniform(0.01, 0.03), size=img_f.shape)
    img_f += noise
    img_f = np.clip(img_f, 0, 1)

    d = -0.04 * dist + size
    td = np.exp(-beta * d)

    # 生成随机的雾颜色
    fog_color = np.random.uniform(0.6, 0.8, size=(3,))  # 随机颜色
    img_f = img_f * td[..., np.newaxis] + fog_color * (1 - td[..., np.newaxis]) * brightness

    img_f = np.clip(img_f * 255, 0, 255).astype(np.uint8)
    return img_f

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有图片路径
    image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))  # 可根据需要更改扩展名

    # 使用 tqdm 显示进度条
    for path in tqdm(image_paths, desc="Processing images", unit="image"):
        image = cv2.imread(path)
        # 生成随机的 beta 和 brightness 参数
        beta = np.random.uniform(0.01, 0.1)  # 可调整范围
        brightness = np.random.uniform(0.4, 0.6)  # 可调整范围

        image_fogv2 = addfogv2(image, beta, brightness)
        output_path = os.path.join(output_folder, os.path.basename(path))
        cv2.imwrite(output_path, image_fogv2)

if __name__ == '__main__':
    # input_folder = r'src_train/'  # 输入文件夹路径
    # output_folder = r'fog_train/'  # 输出文件夹路径
    # process_images(input_folder, output_folder)
    import torch
    print(torch.__version__)
    print(torch.cuda.is_available())
