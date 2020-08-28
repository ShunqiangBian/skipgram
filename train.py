import time
import numpy as np
import tensorflow as tf
import random
import pandas as pd
from collections import Counter

print(tf.__version__)

df=pd.read_csv('data/AutoMaster_Corpus.csv',header=None).rename(columns={0:'text'})
df.head()
text=' '.join(df['text'])
words=text.split(' ')
len(words)
text[:100]

# 定义函数来完成数据的预处理
def preprocess(text, freq=50):
    '''
    对文本进行预处理

    参数
    ---
    text: 文本数据
    freq: 词频阈值
    '''
    # 对文本中的符号进行替换
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace('。', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('，', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()

    # 删除低频词，减少噪音影响
    word_counts = Counter(words)

    trimmed_words = [word for word in words if word_counts[word] > freq]

    return trimmed_words

# 清洗文本并分词
words = preprocess(text)
print(words[:20])

# 构建映射表
vocab = set(words)
len(vocab)
vocab_to_int = {word: index for index, word in enumerate(vocab)}
int_to_vocab = {index: word for index, word in enumerate(vocab)}
int_to_vocab = {index: word for index, word in enumerate(vocab)}
print("total words: {}".format(len(words)))
print("unique words: {}".format(len(set(words))))
# 对原文本进行vocab到int的转换
int_words = [vocab_to_int[w] for w in words]
int_word_counts = Counter(int_words)
t = 1e-3 # t值
threshold = 0.7 # 剔除概率阈值

# 统计单词出现频次
int_word_counts = Counter(int_words)
total_count = len(int_words)
# 计算单词频率
word_freqs = {w: c/total_count for w, c in int_word_counts.items()}
# 计算被删除的概率
prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in int_word_counts}
# 对单词进行采样
train_words = [w for w in int_words if prob_drop[w] < threshold]
drop_words=[int_to_vocab[w] for w in int_words if prob_drop[w] > threshold]
set(drop_words)
len(int_words)
len(train_words)
# int_words

def get_targets(words, idx, window_size=5):
    '''
    获得input word的上下文单词列表

    参数
    ---
    words: 单词列表
    idx: input word的索引号
    window_size: 窗口大小
    '''
    target_window = np.random.randint(1, window_size+1)
    # 这里要考虑input word前面单词不够的情况
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    # output words(即窗口中的上下文单词)
    targets = set(words[start_point: idx] + words[idx+1: end_point+1])
    return list(targets)

train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.compat.v1.placeholder(tf.int32, shape=[None], name='inputs')
    labels = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name='labels')

vocab_size = len(int_to_vocab)
embedding_size = 300 # 嵌入维度

with train_graph.as_default():
    # 嵌入层权重矩阵
    embedding = tf.Variable(tf.compat.v1.random_uniform([vocab_size, embedding_size], -1, 1))
    # 实现lookup
    embed = tf.nn.embedding_lookup(embedding, inputs)
    print(embed)

n_sampled = 1000

with train_graph.as_default():
    softmax_w = tf.Variable(tf.compat.v1.truncated_normal([vocab_size, embedding_size], stddev=0.1))
    softmax_b = tf.Variable(tf.zeros(vocab_size))

    # 计算negative sampling下的损失 nec
    loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, vocab_size)

    cost = tf.reduce_mean(loss)
    optimizer = tf.compat.v1.train.AdamOptimizer().minimize(cost)

with train_graph.as_default():
    #     # 随机挑选一些单词
    #     valid_size = 16
    #     valid_window = 10
    #     # 从不同位置各选8个单词
    #     valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    #     valid_examples = np.append(valid_examples,
    #                                random.sample(range(1000,1000+valid_window), valid_size//2))
    valid_examples = [vocab_to_int['丰田'],
                      vocab_to_int['发动机'],
                      vocab_to_int['刮伤'],
                      vocab_to_int['助力'],
                      vocab_to_int['方向机'],
                      vocab_to_int['雨刮器']]

    valid_size = len(valid_examples)
    # 验证单词集
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # 计算每个词向量的模并进行单位化
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))
    normalized_embedding = embedding / norm
    # 查找验证单词的词向量
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    # 计算余弦相似度
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))

len(train_words)

epochs = 2 # 迭代轮数
batch_size = 2000 # batch大小
window_size = 3 # 窗口大小

with train_graph.as_default():
    saver = tf.compat.v1.train.Saver() # 文件存储

with tf.compat.v1.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    # 添加节点用于初始化所有的变量
    sess.run(tf.compat.v1.global_variables_initializer())

    for e in range(1, epochs+1):
        # 获得batch数据
        batches = get_batches(train_words, batch_size, window_size)
        start = time.time()
        for x, y in batches:

            feed = {inputs: x,
                    labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

            loss += train_loss

            if iteration % 1000 == 0:
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss/1000),
                      "{:.4f} sec/batch".format((end-start)/1000))
                loss = 0
                start = time.time()

            # 计算相似的词
            if iteration % 1000 == 0:
                print('*'*100)
                # 计算similarity
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = int_to_vocab[valid_examples[i]]
                    top_k = 8 # 取最相似单词的前8个
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = 'Nearest to [%s]:' % valid_word
                    for k in range(top_k):
                        close_word = int_to_vocab[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)
                print('*'*100)

            iteration += 1

    save_path = saver.save(sess, "checkpoints/text8.ckpt")
    embed_mat = sess.run(normalized_embedding)

import matplotlib
from matplotlib import font_manager
font=font_manager.FontProperties(fname="TrueType/simhei.ttf")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
viz_words = 1200
tsne = TSNE()
embed_tsne = tsne.fit_transform(embed_mat[:viz_words, :])
fig, ax = plt.subplots(figsize=(50, 50))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color='steelblue')
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7,fontproperties=font)
fig.savefig('auto_master.png')