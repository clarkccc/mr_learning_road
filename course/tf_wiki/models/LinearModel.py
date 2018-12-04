import tensorflow as tf
tf.enable_eager_execution()

# user like model
X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(units=1, kernel_initializer=tf.zeros_initializer(),
                                           bias_initializer=tf.zeros_initializer())

    def call(self, input):
        output = self.dense(input)
        return output


# 这里，我们没有显式地声明 w 和 b 两个变量并写出 y_pred = tf.matmul(X, w) + b 这一线性变换，
# 而是在初始化部分实例化了一个全连接层（ tf.keras.layers.Dense ），并在call方法中对这个层进行调用。
# 全连接层封装了 output = activation(tf.matmul(input, kernel) + bias)
# 这一线性变换+激活函数的计算操作，以及 kernel 和 bias 两个变量。
# 当不指定激活函数时（即 activation(x) = x ），这个全连接层就等价于我们上述的线性变换。
# 顺便一提，全连接层可能是我们编写模型时使用最频繁的层。

# 以下代码结构与前节类似
model = Linear()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)      # 调用模型,将参数定义以及输入的工作放到封装了
        loss =0.5* tf.reduce_sum(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables) #取参数的地方也封装了
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
print(model.variables)
