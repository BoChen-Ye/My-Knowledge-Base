在[[Pipeline#分支预测（Branch Prediction）]]中已经简单介绍过分支预测，本文将讨论更高级的动态分支预测准确率的技术来**降低分支成本**。
# 相关分支预测器
- 相关分支预测器（Correlating predictor）也叫两级预测器（two-level predictor）, 是利用其他分支的行为来进行预测的分支预测器。
- 相关分支预测率高于2位预测器，而需要添加的硬件很少
- 一般情况下，（m，n）预测器利用最近m个分支的行为在$2^m$个分支预测器中进行选择，其中每个分支预测器都是单个分支的n位预测器。
# 竞争预测器tournament predictor
- 竞争预测器采用多个预测器（通常一个全局预测器和一个局部预测器），并使用选择器在他们之间进行选择。
![[Pasted image 20221223143345.png]]
- 全局预测器使用最近的分支历史作为预测器的索引。
- 局部预测器使用分支地址作为索引。
- 将局部分支信息和全局分支历史结合一起的预测器也称为**融合预测器（alloyed predictor）** 或**混合预测器（hybrid predictor）**，竞争预测器是另一种形式的混合或融合预测器。
# 带标签的混合预测器
- 基于PPM（部分匹配预测）的统计压缩算法，也是根据历史来预测未来。
![[Pasted image 20221223145058.png]]