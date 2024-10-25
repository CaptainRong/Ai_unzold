import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks

# 计时开始
start_time = time.time()

# 读取图像
image_path = "balls_2.jpg"
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# CLAHE (自适应直方图均衡化) 用于对比度增强
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(gray_image)

# 高斯模糊
blurred_image = cv2.GaussianBlur(clahe_image, (9, 9), 2)


# 霍夫圆检测 (可以结合分水岭结果)
detected_circles = cv2.HoughCircles(
    blurred_image,
    cv2.HOUGH_GRADIENT,
    dp=2,
    minDist=59,
    param1=200,
    param2=50,
    minRadius=20,
    maxRadius=150
)

# 在原始图像上绘制检测到的圆（NMS前）
output_image_before_nms = image.copy()
radii = []  # 保存圆的半径
detected_circles_nms = []  # 用于存储NMS后的结果
if detected_circles is not None:
    detected_circles = np.round(detected_circles[0, :]).astype("int")

    for (x, y, r) in detected_circles:
        radii.append(r)
        cv2.circle(output_image_before_nms, (x, y), r, (0, 255, 0), 2)

# 显示NMS前检测到的圆
plt.figure(figsize=(6, 6))
plt.title("Detected Circles (Before NMS)")
plt.imshow(cv2.cvtColor(output_image_before_nms, cv2.COLOR_BGR2RGB))
plt.show()

# 非极大值抑制 (NMS) 逻辑，基于圆心距离和半径
def non_max_suppression(circles, overlap_thresh=0.5):
    if len(circles) == 0:
        return []

    # 对圆按半径大小排序
    sorted_indices = np.argsort([r for (x, y, r) in circles])[::-1]
    circles = [circles[i] for i in sorted_indices]

    # 存储NMS后的结果
    keep = []
    while circles:
        # 选择最大的圆并移除
        curr_circle = circles.pop(0)
        keep.append(curr_circle)

        circles = [circle for circle in circles if np.linalg.norm(np.array(curr_circle[:2]) - np.array(circle[:2])) > overlap_thresh * max(curr_circle[2], circle[2])]

    return keep

# 对检测到的圆执行NMS
# detected_circles_nms = non_max_suppression(detected_circles)
detected_circles_nms = detected_circles
# 在原始图像上绘制NMS后的结果
output_image_after_nms = image.copy()
for (x, y, r) in detected_circles_nms:
    cv2.circle(output_image_after_nms, (x, y), r, (0, 255, 0), 2)

# 显示NMS后检测到的圆
plt.figure(figsize=(6, 6))
plt.title("Detected Circles (After NMS)")
plt.imshow(cv2.cvtColor(output_image_after_nms, cv2.COLOR_BGR2RGB))
plt.show()

# 统计NMS后的半径分布并绘制波形图
nms_radii_hist, nms_bin_edges = np.histogram([r for (x, y, r) in detected_circles_nms], bins=20)

plt.figure(figsize=(6, 6))
plt.title("Radius Distribution After NMS")
plt.plot(nms_bin_edges[:-1], nms_radii_hist, color='blue')
plt.xlabel("Radius")
plt.ylabel("Number of Balls")
plt.show()

# 找波峰来划分不同的尺寸类别
peaks, _ = find_peaks(nms_radii_hist, distance=2)

# 打印半径的波峰对应的值
print("Detected radius peaks at:", nms_bin_edges[peaks])

# 计时结束
end_time = time.time()
inference_time = end_time - start_time

# 打印总的小球数和推理时间
total_detected_balls = len(detected_circles_nms)
print(f"Total detected balls after NMS: {total_detected_balls}")
print(f"Inference time: {inference_time:.2f} seconds")

# 在文本文件中记录结果
output_file = "ball_results_after_nms.txt"
with open(output_file, "w") as f:
    f.write(f"Total detected balls after NMS: {total_detected_balls}\n")
    f.write(f"Inference time: {inference_time:.2f} seconds\n")
