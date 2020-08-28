# 使用 TensorFlow 实现 Word2Vec 中的 Skip-Gram 模型

```text
主要步骤：
1. 数据预处理
2. 数据采样
3. 训练数据构造
4. 网络的构建
5. 训练
6. 可视化
```

## 数据预处理
```text
数据预处理过程主要包括：
1. 替换文本中特殊符号并去除低频词
2. 对文本分词
3. 构建语料
4. 单词映射表
5. 首先我们定义一个函数来完成前两步，即对文本的清洗和分词操作。
```

## 数据采样
```text
`skip-gram` 中，训练样本的形式是 `(input word, output word)`，其中 `output word` 是 `input word` 的上下文。为了减少模型噪音并加速训练速度，我们在构造`batch`之前要对样本进行采样，剔除停用词等噪音因素。
对停用词进行采样，例如“你”， “我”以及“的”这类单词进行剔除。剔除这些单词以后能够加快我们的训练过程，同时减少训练过程中的噪音。
我们采用以下公式:
$$ P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}} $$
其中$ t $是一个阈值参数，一般为1e-3至1e-5。
$f(w_i)$ 是单词 $w_i$ 在整个数据集中的出现频次。
$P(w_i)$ 是单词被删除的概率。
>这个公式和论文中描述的那个公式有一些不同
```