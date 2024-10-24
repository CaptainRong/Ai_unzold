import streamlit as st
import cv2
import os

# 设置页面背景颜色和标题颜色
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f5f9;  /* 柔和背景色 */
    }
    .title {
        color: #00252E;  /* 修改标题颜色为#00252E */
        font-size: 2em;  /* 可选：调整字体大小 */
        text-align: center;  /* 可选：居中对齐 */
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
st.markdown("<h1 class='title'>去雾增强车辆重识别系统</h1>", unsafe_allow_html=True)

# 创建左右布局
col1, col2 = st.columns(2)

# 左侧: 查询图像上传
with col1:
    image_file = st.file_uploader("Please upload query image(.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"])
    if image_file is not None:
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
        # 这里应该调用你的重识别模型并进行处理
        # 简单演示：将结果输出到新视频中
        cap = cv2.VideoCapture(video_path)
        output_path = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 示例：在每一帧上画一个矩形框，模拟重识别结果
            cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)
            out.write(frame)

        cap.release()
        out.release()

        # 展示输出视频
        st.video(output_path)
        os.remove(video_path)
