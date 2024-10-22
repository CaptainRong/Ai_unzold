#### By Andxher🐸
🕙2024.10.22 10:18
### Bug&Solution
1.DehazeFormer用于处理大型图片较多，因而通过裁剪为固定大小来处理图像（默认256）然而我们的数据集中大部分图像宽或者高不足256，导致了报错
2.目前解决办法：对于不足256长宽的进行reshape（详见`loader.augument`）。然而事实上造成这种情况出现的根本原因是模型与数据集较不匹配，若出现最终结果与预期不符，其有可能是原因之一。

#### By Andxher🐸
🕚2024.10.21 22:48

### Commit
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



