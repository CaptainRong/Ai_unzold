import streamlit as st
import cv2
import os
import sys
import numpy as np
sys.path.append('/root/workspace/dehaze_picture/src')
sys.path.append("/root/workspace/Dehaze_VeRi/onnx_model")
import dehazeForVideo
import inter
import multiprocessing
from PIL import Image
import subprocess

script_path = '/root/workspace/WebUI/veri_start.sh'

# 设置页面背景颜色
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f5f9;  /* 柔和背景色 */
    }
    .button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# 创建一个保存图像的文件夹
save_dir = "uploaded_images"
save_path = None
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# 页面标题
st.title("去雾重识别系统")

# 创建左右布局
col1, col2 = st.columns(2)

# 左侧: 查询图像上传
with col1:
    image_file = st.file_uploader("Please upload query image(.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"])
    if image_file is not None: 
        # 打开并读取图像
        image = Image.open(image_file)
        
        # 指定文件名（你可以根据需要修改文件名）
        file_name = "uploaded_image.jpg"  # 你可以在这里设置需要的文件名
        
        # 保存图像到新文件夹中
        image.save(os.path.join(save_dir, file_name))
        
        # 显示成功消息
        st.success(f"Image saved successfully at {save_dir}")
        st.image(image_file, caption="Query image", use_column_width=True)


# 右侧: 视频文件上传
with col2:
    video_file = st.file_uploader("Please upload video(.mp4/.avi)", type=["mp4", "avi"])
    if video_file is not None:
        video_path = video_file.name
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        st.video(video_path)

# 底部按钮
if image_file and video_file:
    if st.button("Start ReID"):
      
        # 去雾

        out_dehaze_list = dehazeForVideo.process_video_frame_HaMmer(video_path, "/root/workspace/dehaze_picture/out/output.mp4")

        # 重识别

        # 使用 subprocess.run 执行 shell 脚本
        try:
            result = subprocess.run(['bash', script_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("脚本输出：", result.stdout.decode())
            st.video("/root/workspace/WebUI/out/veri_output.mp4")
        except subprocess.CalledProcessError as e:
            print("脚本执行失败：", e.stderr.decode())
        