import streamlit as st
import cv2
import os
import sys


import numpy as np
from PIL import Image

sys.path.append("../../../common/")
sys.path.append("../")
sys.path.append("../../../common/acllite")
import acl
import acllite_utils as utils
import constants as constants
from acllite_model import AclLiteModel
from acllite_image import AclLiteImage
from acllite_resource import AclLiteResource




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
        print(input_data)
        out = self._model.execute(input_data)
        print(f"\n{out}\n")

        return out

    def post_process_frame(self, infer_output):
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



def init_acl():
    model_path = "/root/workspace/dehaze_picture/model/deploy_vel.om"
    model_width = 512
    model_height = 512

    if 'acl_resource' not in st.session_state:
        print("1")
        st.session_state.acl_resource = AclLiteResource()  # 假设 x1 有一个初始化函数 initialize       
        st.session_state.acl_resource.init()  # 只运行一次 a.init()
        st.write("x1 initialized.")


    if 'single_image_dehaze' not in st.session_state:
        print("2")
        st.session_state.single_image_dehaze = SingleImageDehaze(model_path, model_width, model_height)
        st.session_state.my_class_initialized = False  # 初始化标志
        st.write("MyClass instance created.")

        # ret = st.session_state.single_image_dehaze
        # utils.check_ret("single_image_dehaze init ", ret)
        # if ret != 0:
        #     raise Exception(f"acl.init failed with ret={ret}")
        # st.write("MyClass instance created.")

    if not st.session_state.my_class_initialized:
        print("3")
        ret = st.session_state.single_image_dehaze.init()
        utils.check_ret("single_image_dehaze init ", ret)
        st.session_state.my_class_initialized = True  # 标记初始化已完成
        st.write("MyClass instance initialized.")

def process_video_frame(single_image_dehaze, img):
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
    # 将numpy数组转换为PIL图像对象
    pil_img = Image.fromarray(rgb_frame)
    print(pil_img)
    resized_image = single_image_dehaze.pre_process(pil_img)
    
    # Inference
    result = single_image_dehaze.inference([resized_image, ])
    result = single_image_dehaze.post_process_frame(result)

    return result



# 设置页面背景颜色
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f5f9;
    }
    .button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# 页面标题
st.title("车辆重识别系统")



# 上传视频文件
video_file = st.file_uploader("Please upload video(.mp4/.avi)", type=["mp4", "avi"])

# 初始化 ACL 和模型
init_acl()  # 在处理视频之前调用初始化函数

# 处理并实时显示视频
# 在主逻辑中，保持视频处理部分不变
if video_file is not None:
    video_path = video_file.name
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())

    stframe = st.empty()

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理视频帧
        # processed_frame = process_video_frame(single_image_dehaze, frame)
        processed_frame = process_video_frame(st.session_state.single_image_dehaze, frame)
        processed_frame_rgb = cv2.cvtColor(np.array(processed_frame), cv2.COLOR_BGR2RGB)
        
        stframe.image(processed_frame_rgb, channels="RGB", use_column_width=True)

    cap.release()
    os.remove(video_path)


