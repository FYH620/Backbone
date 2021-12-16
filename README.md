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

#### SqueezeNet  ![](https://img.shields.io/badge/%E8%BD%BB%E9%87%8F%E5%8C%96-SqueezeNet-success)

- 使用 **1*1 卷积**代替 3\*3 卷积，降低参数量；同时**降低通道数**从而降低后续 3\*3 卷积的运算量与参数量
- Squeeze 后使用**多个尺寸的卷积**核进行运算，保留一定的特征信息

#### MobileNet  ![](https://img.shields.io/badge/%E8%BD%BB%E9%87%8F%E5%8C%96-MobileNet-green)

##### MobileNetV1

- 将卷积分解为逐通道卷积与逐点卷积两步，提出了**深度可分离卷积**，降低了运算量
- 使用 ReLU6 作为激活函数，满足了移动端**低精度部署**时的要求，防止输出过大数字无法被覆盖

##### MobileNetV2

- 第一代网络中**各通道特征独立**，加之 ReLU 激活函数，使很多卷积核参数稀疏，梯度更新为 0，学习效果差
- 第二代提出了**反残差模块**，利用了残差网络的基础进行改进，与 ResNet 不同的是其 1*1 卷积进行升维而非降维，这是由于后续的 3\*3 卷积采用了深度可分离卷积，可以减少运算量
- 最终进行逐点相加时**去除了 ReLU6 激活函数**，防止特征被破坏

#### ShuffleNet  ![](https://img.shields.io/badge/%E8%BD%BB%E9%87%8F%E5%8C%96-ShuffleNet-orange)

- 提出了优越的**通道混洗**技术，加强了组卷积过程中的信息流通，增强了特征的表达能力

### Blog

[CVWorld-专注CV领域技术分享](http://cvworld.top/index.php/category/paper-and-code/)

