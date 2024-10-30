import cv2
import numpy as np
import torch

import time
import acl
from mindx.sdk import Tensor
from mindx.sdk import base
from mindx.sdk.base import post
from utils.det_utils import letterbox 
from sklearn.metrics.pairwise import cosine_similarity
import gc
import pickle
import matplotlib.pyplot as plt


# from mx_rec.util.initialize import terminate_config_initializer
DEVICE = 0

model_pth = '/root/workspace/Dehaze_VeRi/onnx_model/model_veri.om'


def init_model():
    base.mx_init()
    model = base.model(modelPath=model_pth, deviceId=DEVICE)
    return model

def deinit_model():
    print("Before del:", any(obj is base for obj in gc.get_objects()))
    acl.rt.reset_device(DEVICE)
    # acl.rt.destroy_context()
    acl.finalize()
    # ret = acl.finalize()    

    global base
    if 'base' in globals():
        del base
    print("del the base")
    gc.collect()

    unreleased_objects = gc.collect()
    # print("After del:", any(obj is base for obj in gc.get_objects()))
    print(f"释放了 {unreleased_objects} 个对象。")
    # terminate_config_initializer()
    return 0

def preprocess(image_pth):
    img_bgr = cv2.imread(image_pth, cv2.IMREAD_COLOR)  # 读入图片
    img, scale_ratio, pad_size = letterbox(img_bgr, new_shape=[256, 256])  # 对图像进行缩放与填充，保持长宽比
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW，将形状转换为 channel first 
    img = np.expand_dims(img, 0).astype(np.float32)  # 得到(1, 3, 640, 640)，即扩展第一维为 batchsize
    img = np.ascontiguousarray(img) / 255.0  # 转换为内存连续存储的数组
    img = Tensor(img) # 将numpy转为转为Tensor类
    return img

def preprocess_from_web(img):
    # file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    # img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = np.array(img)
    # img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_bgr = cv2.imread(image_pth, cv2.IMREAD_COLOR)  # 读入图片
    img, scale_ratio, pad_size = letterbox(img, new_shape=[256, 256])  # 对图像进行缩放与填充，保持长宽比
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW，将形状转换为 channel first 
    img = np.expand_dims(img, 0).astype(np.float32)  # 得到(1, 3, 640, 640)，即扩展第一维为 batchsize
    img = np.ascontiguousarray(img) / 255.0  # 转换为内存连续存储的数组
    img = Tensor(img) # 将numpy转为转为Tensor类
    return img


def postprocess(output, img_pth):
    config_path='/root/workspace/Dehaze_VeRi/onnx_model/utils/VeRi.cfg' # 后处理配置文件
    label_path='/root/Sandra/utils/Resnet_clsindx_2_labels.names' # 类别标签文件

    postprocessor = post.Resnet50PostProcess(config_path=config_path, label_path=label_path) # 获取处理对象
    pred = postprocessor.process([output])[0][0] # 利用SDK接口进行后处理，pred：<ClassInfo classId=... confidence=... className=...>
    confidence = pred.confidence # 获取类别置信度
    className = pred.className # 获取类别名称
    print('{}: {}'.format(className, confidence)) # 打印出结果 
    if img_pth:
        img = cv2.imread(img_pth)
        '''保存推理图片'''
        img_res = cv2.putText(img, f'{className}: {confidence:.2f}', (20, 20), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1) # 将预测的类别与置信度添加到图片
        cv2.imwrite('result/result.png', img_res)
        print('save infer result success')


def sort_frames(frames, query_vec, model, threshold=11):
    
    feature_vecs = []
    print("start sorting...")
    for i, frame in enumerate(frames):
        img = preprocess_from_web(frame)
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
    for i, distance in enumerate(distances):
        frame = frames[i].copy()  # 复制当前帧以便修改
        distance = distances[i]
        label = 1 if distance < threshold else 0
        # print(type(frame))
        frame = np.array(frame)
        # 在左上角绘制种类标记
        cv2.putText(frame, f'Class: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        print(f"{i}/{len(frames)} has been sorted.")
        # 显示距离信息
        print(f"Frame {i}, Distance: {distance}, Class: {label}")
        new_frames.append(frame)
        # inter.deinit()
    return new_frames


# 图片列表转换回视频
def frames_to_video(frames, output_file, fps=30.0, frame_rate=10):
    print(type(frames[0]))

    if not frames:
        print("Error: frames list is empty.")
        return


    img_width, img_height = frames[0].shape[1], frames[0].shape[0]  # 获取图像的宽度和高度


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (img_width, img_height))
    print("converting to video...")
    for i, img in enumerate(frames):
        # 确保 img 是 numpy 数组并转换类型
        if isinstance(img, np.ndarray):
            frame = img.astype(np.uint8)
            print(f"{i}/{len(frames)} has been converted.")

            # 如果是 RGB 格式，需要转换为 BGR 格式
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 写入多帧以达到指定的帧率
            for _ in range(frame_rate):
                out.write(frame)
        else:
            print(f"Warning: Frame {i} is not a numpy array.")

    out.release()
    print(f"Save successfully as {output_file}")


# 去雾任务函数
def run_veri_in_new_process(query_img, dehaze_list):

    model_veri = init_model()
    # st.session_state.sort_model = model
    print(query_img)
    img = preprocess(query_img)

    output = model_veri.infer([img])[0]
    output.to_host()    
    query_vec = np.array(output)
    print(f"query vector get\n{query_vec}\n")

    # For video
    new_frames = sort_frames(dehaze_list, query_vec, model_veri)
    out_new_video_path = "/root/workspace/WebUI/out/veri_output.mp4"

    frames_to_video(new_frames, out_new_video_path)
    return "Hello"

   
def load_processed_images(pickle_file):
    with open(pickle_file, 'rb') as f:
        processed_imgs = pickle.load(f)
    return processed_imgs  


if __name__ == '__main__':
    save_dir = "/root/workspace/WebUI/uploaded_images/uploaded_image.jpg"
    video_dir = "/root/workspace/dehaze_picture/out/output.mp4"
    # if not os.path.exists(query_img):
    #     print(f"File does not exist: {query_img}")
    # img = preprocess(save_dir)
    processed_img = load_processed_images(video_dir)
    run_veri_in_new_process(save_dir, processed_img)







    """
    image_pth = ['/root/workspace/Dehaze_VeRi/onnx_model/data/0002_c007_00085070_0.jpg',
                '/root/workspace/Dehaze_VeRi/onnx_model/data/0003_c019_00019530_0.jpg',
                '/root/workspace/Dehaze_VeRi/onnx_model/data/0004_c001_00043970_0.jpg',
                '/root/workspace/Dehaze_VeRi/onnx_model/data/0005_c002_00075730_0.jpg',
                '/root/workspace/Dehaze_VeRi/onnx_model/data/0720_c001_00073710_1.jpg'
                ]

    quer_img = '/root/workspace/Dehaze_VeRi/onnx_model/data/0720_c003_00069820_0.jpg'

    model_output = []
    base.mx_init()


    model = base.model(modelPath=model_pth, deviceId=DEVICE)
    print(model)

    for path in image_pth: 
        img = preprocess(path)
        output = model.infer([img])[0]
        output.to_host()
        output = np.array(output)
        model_output.append(output[0])

    for i in model_output:
        print(f"output:\t {i}\t type:\t {type(i)}")
    
    img = preprocess(quer_img)
    output = model.infer([img])[0]
    output.to_host()
    query = np.array(output)
    

    # similarities = cosine_similarity(query[0], model_output)
    # most_similar_idx = similarities.argmax()
    # 计算欧氏距离
    distances = np.linalg.norm(model_output- query, axis=1)
    most_similar_idx = np.argmin(distances)
    # 输出最相似图像的索引
    print("最相似的图像索引为：", most_similar_idx)
    print("最小距离为：", distances[most_similar_idx])
    """