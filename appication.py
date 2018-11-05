import MultiLayerModel as mlm
# 创建四层模型中的每一层（第0层为空气层）
l1 = mlm.Layer(1)
l2 = mlm.Layer(2)
l3 = mlm.Layer(3)
# 构建多层模型
ml = mlm.MultiLayerModel([l1, l2, l3])
print('散射矩阵表达式如下：')
print(ml.S)
