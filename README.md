### Commit
#### By Andxher🐸
🕧2024.10.24 12:29

1. 上传了训练得到的模型（`dehaze/saved_models/outdoor/dehazeformer-b_trained.pth`）以及训练模型的部分图片测试结果（`dehaze/result/outdoor`），需要更多测试结果运行`python test.py --model=dehazeformer-b --save_dir=./results/ --dataset=outdoor --exp=outdoor`，其中model.pth需要放在results文件夹下，待去雾图片按照数据集格式放置. 目前从csv的指标(ssim和BCE)对比可以发现总体效果有较明显的提升（也有可能存在过拟合）
2. 上传了GUI（`WebGUI/test.py`），运行请使用streamlit run path/to/code.

### TODO
1. *实现Dehaze和重识别网络的联合训练，并修改loss，使分类/目标检测结果能够反馈给DehazeFormer（实现程度取决于时间够不够）
2. 小球距离测量的代码测试
3. GUI待做
4. 部署及转om模型
5. 报告待做
6. PPT待做

### Bug&Solution
#### By Andxher🐸
🕙2024.10.22 10:18

1.DehazeFormer用于处理大型图片较多，因而通过裁剪为固定大小来处理图像（默认256）然而我们的数据集中大部分图像宽或者高不足256，导致了报错
2.目前解决办法：对于不足256长宽的进行reshape（详见`loader.augument`）。然而事实上造成这种情况出现的根本原因是模型与数据集较不匹配，若出现最终结果与预期不符，其有可能是原因之一。

### Commit
#### By Andxher🐸
🕚2024.10.21 22:48

1. 上传了DehazeFormer源代码和数据集结构（`dehaze/data`）以及第一次测试集最终结果（`dehaze/result`）
2. 上传了生成雾的代码（`dehaze/fog`）

### TODO
1. 目前DehazeFormer的预训练模型不知为何无法进行微调
2. 了解重识别训练方式
3. 实现Dehaze和重识别网络的联合训练，并修改loss，使分类/目标检测结果能够反馈给DehazeFormer
4. 小球距离测量的代码测试
5. GUI待做
6. 部署及转OM模型
7. PPT待做
8. 报告待做



