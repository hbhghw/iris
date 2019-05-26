import tensorflow as tf
import numpy as np


def getData():
    data = []
    labels = []
    labels2int = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    with open('iris.data') as f:
        for _ in range(150):
            line = f.readline().strip().split(',')
            d = [float(i) for i in line[:4]]
            label = [0, 0, 0]
            label[labels2int[line[4]]] = 1
            data.append(d)
            labels.append(label)

    return np.asarray(data), np.asarray(labels)


def buildModel(num_classes=3):
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='inputs')
    targets = tf.placeholder(dtype=tf.int64, shape=[None, num_classes], name='labels')

    net = tf.layers.dense(inputs, 10, activation=tf.nn.relu, use_bias=True)
    net = tf.layers.dense(net, 20, activation=tf.nn.relu, use_bias=True)
    net = tf.layers.dense(net, 3, use_bias=True)
    net = tf.nn.softmax(net, axis=-1, name='outputs')

    losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=net))
    optimizer = tf.train.AdamOptimizer(0.01)
    train_op = optimizer.minimize(losses)

    return inputs, targets, losses, train_op


def train(data, labels):
    inputs, targets, losses, train_op = buildModel()
    trainds = tf.data.Dataset.from_tensor_slices((data, labels)).repeat().shuffle(buffer_size=100).batch(2)
    it = trainds.make_one_shot_iterator()
    next_op = it.get_next()
    saver = tf.train.Saver()
    n = 1000
    with tf.Session() as sess:
        tf.summary.scalar('loss', losses)
        summary_merge = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('logs',sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range(n):
            data_batch, labels_batch = sess.run(next_op)
            _, loss,_summary = sess.run([train_op, losses,summary_merge], feed_dict={inputs: data_batch, targets: labels_batch})
            summary_writer.add_summary(_summary,i)
            if i % 10 == 0:
                print(f"step:{i},loss={loss}")
        saver.save(sess, f'models/iris{n}')


def test(data,labels):
    saver = tf.train.import_meta_graph('models/iris1000.meta')
    sess = tf.Session()
    saver.restore(sess, 'models/iris1000')
    # saver.restore(sess,tf.train.latest_checkpoint('models'))
    inputs = sess.graph.get_tensor_by_name('inputs:0')
    outputs = sess.graph.get_tensor_by_name('outputs:0')
    pred = sess.run(outputs, feed_dict={inputs: data})
    pred = np.argmax(pred, axis=-1)
    labels = np.argmax(labels,axis=-1)
    print(np.mean(pred==labels))


if __name__ == '__main__':
    data, labels = getData()
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:100], indices[100:]
    trainX, trainY, testX, testY = data[train_indices], labels[train_indices], data[test_indices], labels[test_indices]
    train(trainX, trainY)
    test(testX,testY)
