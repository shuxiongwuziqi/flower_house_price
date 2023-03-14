# flower_house_price

本案例使用Flower框架进行联邦学习预测房价，准确度71%，loss为0.68，代码改写自Flower官方案例[sklearn-logreg-mnist](https://github.com/adap/flower/tree/main/examples/sklearn-logreg-mnist)

## 改写重点

主要改写重点是把原来在utils.py中的load_mnist函数改写成load_data

## 特征选取
```python
num_cols = [
  "number of rooms", 
  "security level of the community",
  "residence space",
  # "building space",
  "noise level",
  "waterfront",
  "view",
  "air quality level",
  "aboveground space ",
  # "basement space",
  "building year",
  # "decoration year",
  # "lat",
  # "lng",
  # "total cost"
]
    
cat_cols = [
  "city",
  "zip code",
]
```

选取了两种类型的特征，数字值类型和分类类型(非数字类型)，分类类型用one-hot编码转换，最后使用`StandardScaler`进行规范化。经过测试`building space`和`basement space`等特征对准确度没用贡献，甚至会拉低准确度，所以去除，经纬度特征(通过Google地图API获取)也对训练没有太大帮助。

验证特征对训练的帮助，可以使用single.py文件，这个就是使用传统方式进行训练，能够更快的检验特征的有效性。

## 分析

生成的日志文件my.log可以看到训练细节，analysis.ipynb通过读取该日志文件来分析准确率和loss的变化

