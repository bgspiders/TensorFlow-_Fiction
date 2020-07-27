# coding: utf-8
import tensorflow as tf
import numpy as np
path_to_file = tf.keras.utils.get_file('dou.txt', 'http://d.bxwx666.org/txt/12/11/11.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])
# 词集的长度
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
checkpoint_dir = './training_checkpoints'
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
# model.summary()
def generate_text(model, start_string,num):
    # 要生成的字符个数
    num_generate = num
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    temperature = 1.0
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
def index(num):
    comm=str(generate_text(model, start_string="s",num=num))
    return comm
a=index(num=10000)
print(a)
with open('狗屁不通小说.txt','w+',encoding='utf-8') as f:
        f.write(a)