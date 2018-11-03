# 自注意力与文本分类

本仓库基于自注意力机制实现文本分类。

## 依赖 

- Python 3.5
- Keras 

## 数据集

IMDB影评倾向分类数据集，来自IMDB的25,000条影评，被标记为正面/负面两种评价。影评已被预处理为词下标构成的序列。方便起见，单词的下标基于它在数据集中出现的频率标定，例如整数3所编码的词为数据集中第3常出现的词。

按照惯例，0不代表任何特定的词，而用来编码任何未知单词。

## 用法

### 训练
```bash
$ python imdb_attention.py
```

### 比较结果

|算法|训练时间(每epoch)|Val 准确率|Val Loss|所需Epoch数|
|---|---|---|---|---|
|LSTM|116s|0.8339|0.3815|2|
|双向LSTM|299s|0.8468|0.3475|1|
|CNN|9s|0.8916|0.2713|3|
|CNN LSTM|44s|0.8578|0.3435|2|
|预训练词向量(fasttext)|4s|0.8916|0.2763|7|
|Transformer|8s|0.8432|0.3500|2|

### 结论

最新很火的变形金刚模型(Transformer, 即多头自注意力) 在这个经典的小文本分类数据集上的表现，只是中等。

![image](https://github.com/foamliu/Self-Attention-Keras/raw/master/images/XunlianShijian.PNG)

![image](https://github.com/foamliu/Self-Attention-Keras/raw/master/images/Zhunquelv.PNG)




