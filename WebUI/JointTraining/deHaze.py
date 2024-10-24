import sys
import os

# Assuming the script is inside JointTraining, adjust the path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

# Now use the absolute import
from DehazeFormer.models.dehazeformer import dehazeformer_b

import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torchvision import transforms
import cv2





# 加载训练好的模型
def load_model(model_path, model_name):
    # 创建网络
    network = eval(model_name.replace('-', '_'))()
    network.cuda()

    # 加载模型参数
    state_dict = torch.load(model_path)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # Remove `module.` prefix
        new_state_dict[name] = v
    network.load_state_dict(new_state_dict)
    network.eval()  # 设置为评估模式
    return network


# 对单张图片进行去雾处理
def save_dehaze_image(image_path, model):
    # 读取图片并预处理
    img = cv2.imread(image_path)
    img[:, :, ::-1].astype('float32') / 255.0
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化 [-1, 1]
    ])
    input_img = transform(img).unsqueeze(0).cuda()  # 增加batch维度

    # 模型推理
    with torch.no_grad():
        output = model(input_img).clamp_(-1, 1)

    # 反归一化并将结果转换为图片
    output = output * 0.5 + 0.5  # 将 [-1, 1] 转换为 [0, 1]
    output_img = output.squeeze(0).cpu().numpy()
    output_img = np.transpose(output_img, (1, 2, 0))  # [C, H, W] -> [H, W, C]
    output_img = (output_img * 255).astype(np.uint8)  # 转换为8位图像

    # 使用 OpenCV 保存结果
    output_path = image_path.replace(".jpg", "_dehazed.jpg")
    cv2.imwrite(output_path, output_img)
    return output_path


def process_hazy(img, model):
    # 读取图片并预处理
    img[:, :, ::-1].astype('float32') / 255.0
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化 [-1, 1]
    ])
    input_img = transform(img).unsqueeze(0).cuda()  # 增加batch维度

    # 模型推理
    with torch.no_grad():
        output = model(input_img).clamp_(-1, 1)

    # 反归一化并将结果转换为图片
    output = output * 0.5 + 0.5  # 将 [-1, 1] 转换为 [0, 1]
    output_img = output.squeeze(0).cpu().numpy()
    # output_img = np.transpose(output_img, (1, 2, 0))  # [C, H, W] -> [H, W, C]
    # output_img = (output_img * 255).astype(np.uint8)  # 转换为8位图像
    return output_img


if __name__ == '__main__':
    # 你要去雾的图片路径
    image_path = 'data/hazy2.jpg'

    # 处理单张图片
    result_path = save_dehaze_image(image_path, model)
    print(f"去雾处理完成，结果保存为: {result_path}")
