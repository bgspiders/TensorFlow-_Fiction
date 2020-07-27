# coding: utf-8

import tensorflow as tf
import numpy as np
import os
import time
# 后面跟文件的地址，也可以先用我这个的地址
path_to_file = tf.keras.utils.get_file('dou.txt', 'http://d.bxwx666.org/txt/12/11/11.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])
# 设定每个输入句子长度的最大值
seq_length = 100
examples_per_epoch = len(text) // seq_length
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
dataset = sequences.map(split_input_target)
# 批大小
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
vocab_size = len(vocab)
# 嵌入的维度
embedding_dim = 256
# RNN 的单元数量
rnn_units = 1024
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model
model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
model.summary()
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
example_batch_loss = loss(target_example_batch, example_batch_predictions)
model.compile(optimizer='adam', loss=loss)
#训练次数
EPOCHS = 50
# 模型保存至的目录
checkpoint_dir = './training_checkpoints'
# 模型保存的文件名
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
# 查看模型
model.summary()
def generate_text(model, start_string):
    num_generate = 1000
    # 将起始字符串转换为数字（向量化）
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    # 空字符串用于存储结果
    text_generated = []
    # 低温度会生成更可预测的文本
    # 较高温度会生成更令人惊讶的文本
    # 可以通过试验以找到最好的设定
    temperature = 1.0
    # 这里批大小为 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # 删除批次的维度
        predictions = tf.squeeze(predictions, 0)
        # 用分类分布预测模型返回的字符
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))
#start_string开头的单词
print(str(generate_text(model, start_string="s")))