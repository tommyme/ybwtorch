## 和我一起写一个自己专属的深度学习工具库吧

> 这个库中有图像分类和目标价测的模块，可以直接拿来用
>
> 图像分类默认用efficientnet-b4
>
> 默认16G显存，如果你有32G可以上b5
>
### 说明
1. 需要根据数据集重写data_pps
2. 根据你自己的需求改变config中的超参数
3. 想实现更多功能的小伙伴可以提出你宝贵的建议

### tricks

- [x] bagging
- [x] autoaugment
- [x] ranger
- [x] mean std
- [x] label smooth
- [x] ReduceLROnPlateau
- [x] cbam模块
- [x] sample weight
- [x] 辅助分类器

### functions

- [x] 混淆矩阵
- [x] 存下验证集上的最优模型
- [x] debug模式
- [x] tqdm打印训练日志
- [x] tpu训练
- [ ] tensorboard查看训练过程
- [ ] 画损失函数的曲线

