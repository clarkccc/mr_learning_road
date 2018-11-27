import tensorflow as tf
tf.enable_eager_execution()

a = tf.constant(1)
b = tf.constant(1)
c = tf.add(a, b)    # 也可以直接写 c = a + b，两者等价

print(c)

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B) #矩阵操作

print(C)

# 自动求导机制

x = tf.get_variable('x', shape=[1], initializer=tf.constant_initializer(3.)) #声明变量并赋予初始值
with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)
y_grad = tape.gradient(y, x)        # 计算y关于x的导数
print([y.numpy(), y_grad.numpy()])
