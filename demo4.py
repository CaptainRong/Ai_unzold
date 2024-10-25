import cv2
import numpy as np
from collections import deque

# 初始化参数
frame_window = 10  # 设置帧窗口大小
radius_threshold = [1, 50]  # 半径阈值范围
ball_count_buffer = deque(maxlen=frame_window)  # 用于累积每帧小球数量

def detect_balls_in_frame(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)
    blurred_image = cv2.GaussianBlur(clahe_image, (9, 9), 2)

    detected_circles = cv2.HoughCircles(
        blurred_image,
        cv2.HOUGH_GRADIENT,
        dp=2,
        minDist=59,
        param1=200,
        param2=50,
        minRadius=radius_threshold[0],
        maxRadius=radius_threshold[1]
    )

    circles = []
    if detected_circles is not None:
        detected_circles = np.round(detected_circles[0, :]).astype("int")
        circles = [(x, y, r) for (x, y, r) in detected_circles]

    # 使用非极大值抑制
    return non_max_suppression(circles)

# 非极大值抑制函数
def non_max_suppression(circles, overlap_thresh=0.5):
    if len(circles) == 0:
        return []

    sorted_indices = np.argsort([r for (x, y, r) in circles])[::-1]
    circles = [circles[i] for i in sorted_indices]
    keep = []

    while circles:
        curr_circle = circles.pop(0)
        keep.append(curr_circle)
        circles = [circle for circle in circles
                   if np.linalg.norm(np.array(curr_circle[:2]) - np.array(circle[:2]))
                   > overlap_thresh * max(curr_circle[2], circle[2])]
    return keep

# 摄像头处理主函数
def process_camera():
    cap = cv2.VideoCapture(0)  # 0表示默认摄像头

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break

        detected_balls = detect_balls_in_frame(frame)
        ball_count_buffer.append(len(detected_balls))

        # 取出前几帧的数量，计算滑动平均
        stable_count = int(np.mean(ball_count_buffer))

        # 可视化检测
        for (x, y, r) in detected_balls:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)

        # 在帧上显示稳定的小球计数
        cv2.putText(frame, f'Stabilized Ball Count: {stable_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Ball Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

process_camera()