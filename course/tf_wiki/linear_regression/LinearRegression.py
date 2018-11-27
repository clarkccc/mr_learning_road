import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = X_raw
y = y_raw
# 这一步称为归一化操作,如果不是这部操作,能计算出来,但是量纲很难计算,使得收敛变得困难
# 主要影响的是学习速率以及初始值,如果归一化后可以尽情取0,不然很难收敛

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

print(X,y)
X = tf.constant(X)
y = tf.constant(y)

a = tf.get_variable('a', dtype=tf.float32, initializer=tf.constant(1000.0))
b = tf.get_variable('b', dtype=tf.float32, initializer=tf.constant(12000.0-2000*2013.0))
variables = [a, b]

num_epoch = 10000
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.95/1e7)
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
        # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
print(a*X+b-y)
