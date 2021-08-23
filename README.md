数据集来自以下地址：

https://github.com/yl-1993/learn-to-cluster

https://github.com/Zhongdao/gcn_clustering/


在训练前需要先使用显卡制作KNN(K近邻)

```angular2html
python utils/knn.py
```

训练参数说明
训练cdp需指定的参数

|   参数   |  含义    |
| ---- | ---- |
|   --train   |  训练    |
|   --cdp   |   使用cdp模式   |
|   --max_size   |   类别的最大实例数   |
|   --step   |   阈值增长的快慢   |

训练dbscan需指定的参数

|   参数   |  含义    |
| ---- | ---- |
|   --train   |  训练    |
|   --min_sample   |   最小样本   |
|   --r   |   指的是r1,这里为了增强模型的鲁棒性，r2=r1+0.1   |

参数设置较少，继续细化调整参数可能得到更好结果。

