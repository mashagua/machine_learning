import tensorflow as tf
import numpy as np
import pickle
with open(r'G:\machine_learning\DSSM\data.pkl', 'rb') as f:
    train_set, test_set, word2id, id2word = pickle.load(f)

embedding_dim = 200
max_sen_len = 15
lr = 0.001
epoch_num = 70
batch_size = 200


def random_embedding(id2word, embedding_dim):
    embedding_mat = np.random.uniform(-0.25,
                                      0.25, (len(id2word), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


embeddings = random_embedding(id2word, embedding_dim)

def get_batch(data, batch_size, shuffle=False):
    '''
    :param data:
    :param batch_size:
    :param shuffle:
    :return:
    '''
    if shuffle:
        np.random.shuffle(data)
    for i in range(len(data) // batch_size):
        data_size = data[i * batch_size:(i + 1) * batch_size]
        s1_data, s2_data, label_data = [], [], []
        for (s1_set, s2_set, y_set) in data_size:
            s1_data.append(s1_set)
            s2_data.append(s2_set)
            label_data.append(y_set)
            yield np.array(s1_data), np.array(s2_data), np.reshape(label_data,(-1, 1))


# 定义一张图
graph_mhd = tf.Graph()
with graph_mhd.as_default():
    left_input = tf.placeholder(
        tf.int32,
        shape=[
            None,
            max_sen_len],
        name='left_input')
    right_input = tf.placeholder(
        tf.int32,
        shape=[
            None,
            max_sen_len],
        name='right_input')
    labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels')
    # 因为做交叉熵，所以变成nONE,1的维度,若每层都dropout,有一些没参与训练，并且后面会越来越小，那么就变为原来的1.2倍
    dropout_p1 = tf.placeholder(tf.float32, shape=(), name='dropout')

    with tf.variable_scope("embeddings"):
        # 矩阵定义成了权重
        _word_embeddings = tf.Variable(embeddings,
                                       dtype=tf.float32,
                                       trainable=True,
                                       name='embedding_matrix'
                                       )
        # 根据每个[[1,5,20,6]，【】]找到对应的
        left_embeddings = tf.nn.embedding_lookup(
            params=_word_embeddings, ids=left_input, name='left_embeddings')
        right_embeddings = tf.nn.embedding_lookup(
            params=_word_embeddings, ids=right_input, name='right_embeddings')
        left_embeddings = tf.nn.dropout(left_embeddings, dropout_p1)
        right_embeddings = tf.nn.dropout(right_embeddings, dropout_p1)
    ##
    with tf.variable_scope("one_layer_bilstm"):
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=50)
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=50)
        # 前向后向拼接在一起变成双向LSTM,
        (left_output_fw_seq, left_output_bw_seq), left_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, left_embeddings, dtype=tf.float32)
        # 左边和右边用的同一套词向量，同一个cell_fw,cell_bw,因为要算距离，必须使用同一个计算空间
        left_bi_output = tf.concat(
            [left_states[0].h, left_states[1].h], axis=1)
        # 第一个是output输出，right_states是隐藏神经元memory的输出，看双向lstm源码
        (right_output_fw_seq, right_output_bw_seq), right_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, right_embeddings, dtype=tf.float32)
        right_bi_output = tf.concat(
            [right_states[0].h, right_states[1].h], axis=1)

    with tf.variable_scope("Similarity_calculation_layer"):
        def cosine_dist(input1, input2):
            pooled_len_1 = tf.sqrt(tf.reduce_sum(input1 * input1, 1))
            pooled_len_2 = tf.sqrt(tf.reduce_sum(input2 * input2, 1))
            pooled_mul_12 = tf.reduce_sum(input1 * input2, 1)
            score = tf.div(
                pooled_mul_12,
                pooled_len_1 *
                pooled_len_2 +
                1e-8,
                name='score')
            return score

        def manhattan_dist(input1, input2):
            score = tf.exp(-tf.reduce_sum(tf.abs(input1 - input2), 1))
            return score

        def multiply(input1, input2):
            score = tf.multiply(input1, input2)
            return score

        def substract(input1, input2):
            score = tf.abs(input1 - input2)
            return score

        def maximum(input1, input2):
            s1 = multiply(input1, input1)
            s2 = multiply(input2, input2)
            score = tf.maximum(s1, s2)
            return score
        # [batch_size,],是cosine值
        cos = cosine_dist(left_bi_output, right_bi_output)

        man = manhattan_dist(left_bi_output, right_bi_output)
        mul = multiply(left_bi_output, right_bi_output)
        # [batch_size,100]
        sub = substract(left_bi_output, right_bi_output)
        maxium = maximum(left_bi_output, right_bi_output)

        last_list_layer = tf.concat([mul, sub, maxium], 1)
        last_drop = tf.nn.dropout(last_list_layer, 0.8)
        dense_layer1 = tf.layers.dense(last_drop, 16, activation=tf.nn.relu)
        dense_layer2 = tf.layers.dense(last_drop, 24, activation=tf.nn.sigmoid)
        output = tf.concat([dense_layer1, dense_layer2, tf.expand_dims(
            cos, -1), tf.expand_dims(man, -1)], 1)

    with tf.variable_scope("classification"):
        # dnn层数越多效果越好
        output = tf.layers.dense(output, 32)
        logits = tf.layers.dense(output, 1)

    with tf.variable_scope("loss"):
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=logits))

    # 选择优化器
    with tf.variable_scope("training_step"):
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    # 是一个非凸函数，却用凸优化来寻找
    with tf.variable_scope("evaluation"):
        pred_rate = tf.sigmoid(logits, name='sig')
        pred = tf.cast(tf.greater(pred_rate, 0.5), tf.float32)
        accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    pred,
                    labels),
                tf.float32),
            name="accuracy")
        true = tf.reshape(labels, (-1,))
        pred = tf.reshape(pred, (-1,))
        epsilon = 1e-7
        cm = tf.contrib.metrics.confusion_matrix(true, pred, num_classes=2)
        precision = tf.cast(
            cm[1][1] / tf.reduce_sum(cm[:, 1]), tf.float32, name="precision")
        recall = tf.cast(
            cm[1][1] /
            tf.reduce_sum(
                cm[1],
                axis=0),
            tf.float32,
            name="recall")
        f1_score = tf.cast((2 *
                            precision *
                            recall /
                            (precision +
                             recall +
                             epsilon)), tf.float32, name="f1_score")
      
with tf.Session(graph=graph_mhd) as sess:
    if True:
        saver = tf.train.Saver(tf.global_variables())
        init = tf.global_variables_initializer()
        sess.run(init)
        global_nums = 0
        for epoch in range(epoch_num):
            for s1, s2, y in get_batch(train_set, batch_size, shuffle=True):
                _, l, acc, p, r, f, cmm = sess.run(
                    [train_op, loss,accuracy,precision,recall,f1_score,cm], {
                    left_input: s1,
                    right_input: s2,
                    labels : y,
                    dropout_p1: 0.8
                })
                global_nums += 1
                if global_nums % 50 == 0:
                    print(cmm)
                    # saver.save(sess, '../model_save/model.ckpt', global_step=global_nums)
                    print(
                        'train: epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4} , precision {}, recall: {:.4}, fbeta_score: {:.4} '.format(epoch , global_nums,
                                                                                          l, acc,p, r, f))


                if global_nums % 200 == 0:
                    print('-----------------valudation---------------')
                    s1, s2, y = next(get_batch(test_set, np.shape(test_set)[0],  shuffle=True))
                    l, acc, p, r, f,cmm = sess.run(
                        [ loss, accuracy, precision, recall, f1_score,cm], {
                            left_input: s1,
                            right_input: s2,
                            labels: y,
                            dropout_p1: 0.8
                        })
                    print(cmm)
                    print(
                        'valudation: epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4} , precision {}, recall: {:.4}, fbeta_score: {:.4} '.format(
                            epoch, global_nums,
                            l, acc, p, r, f))
                    print('-----------------valudation---------------')
        s1, s2, y = next(get_batch(test_set, np.shape(test_set)[0], shuffle=True))
        l, acc, p, r, f, cmm = sess.run(
                    [loss, accuracy, precision, recall, f1_score, cm], {
                        left_input: s1,
                        right_input: s2,
                        labels: y,
                        dropout_p1: 0.8
                    })
        print(cmm)
        saver.save(sess, 'save_model/model.ckpt', global_step=global_nums)
        print('test: loss: {:.4}, accuracy: {:.4} , precision {}, recall: {:.4}, fbeta_score: {:.4} '.format(
                        l, acc, p, r, f))