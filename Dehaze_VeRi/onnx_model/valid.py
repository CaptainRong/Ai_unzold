import onnxruntime as ort
# 加载onnx模型
ort_session = ort.InferenceSession('model_best.onnx')

input_name = ort_session.get_inputs()[0].name

print(input_name)
