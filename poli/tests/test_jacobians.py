import tensorflow as tf
import numpy as np

if __name__ == '__main__':

    batch_size = 30
    # A = tf.get_variable(name='A', shape=(4,3), dtype=tf.float32)
    # x = tf.placeholder(tf.float32, shape=(3, batch_size), name='x')
    # R = tf.placeholder(tf.float32, shape=(4, batch_size), name='R')
    # y = tf.matmul(A, x)

    # gradients = [tf.gradients(ys=y[i], xs=A, grad_ys=R[i]) for i in range(4)]

    # A_val = np.random.rand(4,3)
    # x_val = np.random.rand(3,batch_size)
    # R_val = np.random.rand(4,batch_size)


    A = tf.get_variable(name='A', shape=(4,3), dtype=tf.float32)
    x = tf.placeholder(tf.float32, shape=(batch_size, 3), name='x')
    R = tf.placeholder(tf.float32, shape=(batch_size, 4), name='R')
    y = tf.matmul(x, A, transpose_b=True)

    gradients = tf.gradients(ys=tf.unstack(y, axis=1), xs=A, grad_ys=tf.unstack(R, axis=1))

    A_val = np.random.rand(4,3)
    x_val = np.random.rand(batch_size, 3)
    R_val = np.random.rand(batch_size, 4)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.assign(A, A_val))