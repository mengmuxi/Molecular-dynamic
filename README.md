# Molecular-dynamic
我们利用深度学习算法来帮助描述自驱动粒子的集群行为，将基于图网络搭建的模型用于集群动力学系统，该模型实现了对不同视野角下以及 的长期预测，并且得到了很好的效果。


数据集

数据集由Generate dataset\phi.py 生成


训练
1. 数据预处理：Graph Network Prediction\graph_model.py
将数据集处理为图结构
2. 训练：Graph Network Prediction\train.py
搭建图网络模型，训练模型
3. 加载训练数据集：Graph Network Prediction\train_binary.py
4. 预测：Graph Network Prediction\train_test.py
