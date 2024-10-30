"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2021-02-02 09:44:13
MODIFIED: 2021-02-22 09:44:13
"""
import sys
import os
import numpy as np
from PIL import Image
import pickle

sys.path.append("../../../common/")
sys.path.append("../")
sys.path.append("../../../common/acllite")
sys.path.append("/root/workspace/Dehaze_VeRi/onnx_model")

import acl
import acllite_utils as utils
import constants as constants
from acllite_model import AclLiteModel
from acllite_image import AclLiteImage
from acllite_resource import AclLiteResource
import cv2
import inter


class SingleImageDehaze(object):
    """
    Class for SingleImageDehaze
    """
    def __init__(self, model_path, model_width, model_height):
        self._model_path = model_path
        self._model_width = model_width
        self._model_height = model_height
        self._img_width = 0
        self._img_height = 0
        self._model = None

    @staticmethod
    def sigmoid(x):
        """
        sigmod function
        """
        return 1. / (1 + np.exp(-x))

    def init(self):
        """
        Initialize
        """
        # Load model
        self._model = AclLiteModel(self._model_path)

        return constants.SUCCESS

    def pre_process(self, im):
        """
        image preprocess
        """
        self._img_width = im.size[0]
        self._img_height = im.size[1]
        im = im.resize((512, 512))
        # hwc
        img = np.array(im)
        img = img / 127.5 - 1.
        
        # rgb to bgr
        img = img[:, :, ::-1]
        img = img.astype("float16")
        return img 

    def inference(self, input_data):
        """
        model inference
        """
        # print(input_data)
        out = self._model.execute(input_data)
        # print(out)

        return out

    def post_process(self, infer_output, image_name):
        """
        Post-processing, analysis of inference results
        """
        np.array(infer_output[0])
        result_image = np.reshape(infer_output[0], (512, 512, 3))
        result_image = result_image[:, :, ::-1]
        result_image = np.clip((result_image + 1.) / 2. * 255., 0, 255).astype(np.uint8)
        result_image = Image.fromarray(result_image)
        result_image = result_image.resize((self._img_width, self._img_height))
        result_image.save('/root/Dehaze_VeRi/dehaze_picture/out/out_' + image_name)

    def post_process_video(self, infer_output):
        """
        Post-processing, analysis of inference results
        """
        np.array(infer_output[0])
        result_image = np.reshape(infer_output[0], (512, 512, 3))
        result_image = result_image[:, :, ::-1]
        result_image = np.clip((result_image + 1.) / 2. * 255., 0, 255).astype(np.uint8)
        result_image = Image.fromarray(result_image)
        result_image = result_image.resize((self._img_width, self._img_height))
        return result_image


def video_to_frames(video_file, frame_rate=10):
    # 初始化一个视频捕捉对象
    cap = cv2.VideoCapture(video_file)
    
    # 初始化一个列表来保存帧
    frames = []
    
    # 检查是否成功打开视频文件
    if not cap.isOpened():
        print("Error opening video file")
        return None
    
    i = frame_rate
    # 循环读取视频的每一帧
    while True:
        
        ret, frame = cap.read()
        if not ret:
            break
        if i == frame_rate:
            # 将OpenCV的BGR格式转换为RGB格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 将numpy数组转换为PIL图像对象
            pil_img = Image.fromarray(rgb_frame)
            frames.append(pil_img)
            i = 0
        i += 1
    
    # 释放视频捕捉对象
    cap.release()
    
    return frames


def sort_frames(frames, query_vec, threshold=20):
    model = inter.init_model()
    feature_vecs = []
    print("start sorting...")
    for frame in frames:
        img = inter.preprocess_from_web(frame)
        output = model.infer([img])[0]
        output.to_host()
        feature_vec = np.array(output)
        feature_vecs.append(feature_vec)
        print(f"{i}/{len(frames)} has been calculated.")
    # 计算欧氏距离并排序
    distances = [np.linalg.norm(feature_vec - query_vec) for feature_vec in feature_vecs]
    sorted_indices = np.argsort(distances)

    new_frames = []
    # 绘制种类标记
    for i in sorted_indices:
        frame = frames[i].copy()  # 复制当前帧以便修改
        distance = distances[i]
        label = 1 if distance < threshold else 0
        
        # 在左上角绘制种类标记
        cv2.putText(frame, f'Class: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        print(f"{i}/{len(frames)} has been sorted.")
        # 显示距离信息
        print(f"Frame {i}, Distance: {distance}, Class: {label}")
        new_frames.append(frame)
        inter.deinit()
    return new_frames
    




# 图片列表转换回视频
def frames_to_video(frames, output_file, fps=30.0, frame_rate=10):

    img_width, img_height = frames[0].size

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (img_width, img_height))
    print("converting to video...")
    for i, img in enumerate(frames):
        # 将PIL图像转换为numpy数组
        frame = np.array(img)
        print(f"{i}/{len(frames)} has been converted.")
        # 如果是RGB格式，需要转换为BGR格式
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for i in range (0, frame_rate):
            out.write(frame)

    out.release()
    print(f"save successfully as {output_file}")


def process_video_frame(video, save_pth, query_vec):
    acl_resource = AclLiteResource()
    acl_resource.init()
    src_path = os.path.realpath(__file__).rsplit("/", 1)[0]
    model_path = "/root/workspace/dehaze_picture/model/deploy_vel.om"
    model_width = 512
    model_height = 512
    frame_rate = 30
    # With picture directory parameters during program execution
    # if len(sys.argv) != 2:
    #     print("The App arg is invalid")
    #     exit(1)

    single_image_dehaze = SingleImageDehaze(model_path, model_width, model_height)
    ret = single_image_dehaze.init()
    utils.check_ret("single_image_dehaze init ", ret)

    images_list = video_to_frames(video, frame_rate=frame_rate)
    
    total_num = len(images_list)
    print(f"{total_num} imgs in total.")
    processed_imgs = []
    # Create a directory to save inference results
    if not os.path.isdir(os.path.join(src_path, "../out")):
        os.mkdir(os.path.join(src_path, "../out"))

    for i, im in enumerate(images_list):
        # read image
        # im = Image.open(image_file)
        # print(im)
        # Preprocess the picture 
        resized_image = single_image_dehaze.pre_process(im)
        
        # Inference
        result = single_image_dehaze.inference([resized_image, ])
        print(f"{i+1}/{total_num} image has been processed.")
        # Post-processing
        processed_imgs.append(single_image_dehaze.post_process_video(result))
    sorted_frames = sort_frames(processed_imgs, query_vec, threshold=50)
    # sorted_frames = processed_imgs
    frames_to_video(sorted_frames, save_pth, frame_rate=frame_rate)
    return save_pth

def process_video_frame_HaMmer(video, save_pth):
    acl_resource = AclLiteResource()
    acl_resource.init()
    src_path = os.path.realpath(__file__).rsplit("/", 1)[0]
    model_path = "/root/workspace/dehaze_picture/model/deploy_vel.om"
    model_width = 512
    model_height = 512
    frame_rate = 30
    # With picture directory parameters during program execution
    # if len(sys.argv) != 2:
    #     print("The App arg is invalid")
    #     exit(1)

    single_image_dehaze = SingleImageDehaze(model_path, model_width, model_height)
    ret = single_image_dehaze.init()
    utils.check_ret("single_image_dehaze init ", ret)
    
    images_list = video_to_frames(video, frame_rate=frame_rate)
    
    total_num = len(images_list)
    print(f"{total_num} imgs in total.")
    processed_imgs = []
    # Create a directory to save inference results
    if not os.path.isdir(os.path.join(src_path, "../out")):
        os.mkdir(os.path.join(src_path, "../out"))

    for i, im in enumerate(images_list):
        # read image
        # im = Image.open(image_file)
        # print(im)
        # Preprocess the picture 
        resized_image = single_image_dehaze.pre_process(im)
        
        # Inference
        result = single_image_dehaze.inference([resized_image, ])
        print(f"{i+1}/{total_num} image has been processed.")
        # Post-processing
        processed_imgs.append(single_image_dehaze.post_process_video(result))
    # sorted_frames = sort_frames(processed_imgs, query_vec, threshold=50)
    # sorted_frames = processed_imgs
    # frames_to_video(sorted_frames, save_pth, frame_rate=frame_rate)
    with open(save_pth, 'wb') as f:
        pickle.dump(processed_imgs, f)
    return processed_imgs




def display_video(video_file):
    # 使用cv2.VideoCapture读取视频文件
    cap = cv2.VideoCapture(video_file)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    # 循环读取视频帧
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        # 如果读取成功，显示帧
        if ret:
            cv2.imshow('Video', frame)
            
            # 等待按键，如果按下'q'键，则退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # 读取失败，退出循环
            break
    
    # 释放视频捕捉对象并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    save_pth = process_video_frame("/root/workspace/328554195-1-208.mp4", "/root/workspace/dehaze_picture/out/output.mp4")
    # display_video(save_pth)
