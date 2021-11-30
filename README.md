## Backbone 复现

#### AlexNet  ![](https://img.shields.io/badge/-AlexNet-brightgreen)

- **首次**将深度卷积网络应用于图像分类并取得优良的效果

![](https://cdn.jsdelivr.net/gh/FYH620/PicGo-Use/imgs/alexnet.png)

#### VGG16  ![](https://img.shields.io/badge/-VGG16-yellow)

- 多个 **3*3 卷积核**堆叠，提高感受野的同时提升了模型的非线性表达
- 小卷积核参数量少
- 适当加深了网络

![](https://cdn.jsdelivr.net/gh/FYH620/PicGo-Use/imgs/vgg16.png)

#### GoogleNet  ![](https://img.shields.io/badge/Inception%20V1-GoogleNet-orange)

##### Inception V1

- **通道级联**，得到更好的图像表征，使网络有机会可选择最佳特征
- 1*1 卷积核实现**降维**，使网络更加轻量化
- 全连接层前采用**全局平均池化**，融合了深度网络内的语义信息，极大地降低了模型的参数量
- 定义**辅助损失函数**

![](https://cdn.jsdelivr.net/gh/FYH620/PicGo-Use/imgs/googlenet.png)

##### Inception V2

- 提出了 **Batch Normalization** ，缓解了深度网络梯度消失的问题
- 用 1*n 卷积与 n\*1 卷积替换 n\*n 卷积，降低了参数运算量

##### Inception V3

- RMSProp 优化器的使用
- 标签平滑

##### Inception V4

- 与残差网络的有机融合

#### ResNet 50  ![](https://img.shields.io/badge/-ResNet%2050-red)

- **优化残差映射，使网络可以跳跃连接**
- 加深了网络，缓解了深层网络的梯度消失，降低了深层网络的白噪声
- 成为**许多专用于目标检测与分割任务的基础模块**

![](https://cdn.jsdelivr.net/gh/FYH620/PicGo-Use/imgs/resnet1.png)

#### DenseNet  ![](https://img.shields.io/badge/-DenseNet-lightgrey)

- **网络连接密集，每一层均接受后层的梯度**
- **通道级联使得特征被多次复用**，参数更少且高效

![](https://cdn.jsdelivr.net/gh/FYH620/PicGo-Use/imgs/desnet3.png)

#### FPN  ![](https://img.shields.io/badge/-FPN-blue)

- **专用于目标检测的 Backbone**
- **融合了深层次的语义信息与浅层网络的高分辨率**，缓解了检测任务的多尺度问题
- 上采样后继续采用卷积得到输出，消除了上采样过程的重叠效应，平滑了最终的特征图
- 根据不同尺度的输出配合不同的 ROI 区域进行选择

![](https://cdn.jsdelivr.net/gh/FYH620/PicGo-Use/imgs/fpn2.png)

#### DetNet  ![](https://img.shields.io/badge/-DetNet-yellowgreen)

- **空洞卷积**的引入使模型兼具高分辨率与高感受野
- 空洞卷积避免了上采样，降低了计算量
- 融合了多个通道的特征

![](https://cdn.jsdelivr.net/gh/FYH620/PicGo-Use/imgs/detnet4.png)

### Blog

[CVWorld-专注CV领域技术分享](http://cvworld.top/index.php/category/paper-and-code/)

