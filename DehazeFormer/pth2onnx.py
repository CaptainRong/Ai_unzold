import torch
# from models import CNN_face
# from gan_network import _netG
from models.dehazeformer import dehazeformer_b

def load_model(pth='model_net.pth'):
    '''
    :param pth: pth文件路径
    model 是你的模型
    :return:
    '''
    model = dehazeformer_b()
    # model = CNN_face.FaceCNN()  # 创建模型实例
    # model = _netG(nz=100, ngf=64, nc=3)
    # checkpoint = torch.load('your_model.pth', map_location=torch.device('cpu'))

    model.load_state_dict(torch.load(pth))  # 加载保存的参数
    model.eval()  # 设置为评估模式
    return model


def export_to_onnx(model, output_file="model_net.onnx", input_size=(128, 1, 48, 48)):
    """

    :param model: model的框架，用load_model得到
    :param output_file: 输出路径
    :param input_size: 输入大小
    :return: none
    """
    # 创建一个假的输入张量，形状要与实际输入一致
    dummy_input = torch.randn(input_size)
    # 导出为 ONNX 格式
    torch.onnx.export(
        model,                     # 要导出的模型
        dummy_input,               # 输入的示例张量
        output_file,               # 输出文件的路径
        export_params=True,        # 是否导出模型参数
        opset_version=11,          # ONNX 算子的版本，选择支持度较好的 opset 版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['input'],     # 输入的名称（可选）
        output_names=['output'],   # 输出的名称（可选）
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 动态维度支持
    )
    print(f"模型已成功导出为 {output_file}")


if __name__ == '__main__':
    model = load_model("saved_models/dehazeformer-b_trained.pth")  # 加载模型
    export_to_onnx(model, output_file="dehaze_.onnx", input_size=(256, 256, 3, 1))  # 导出为 ONNX

