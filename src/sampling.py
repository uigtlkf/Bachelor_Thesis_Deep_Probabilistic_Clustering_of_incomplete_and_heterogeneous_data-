import tensorflow as tf
from tensorflow.keras.layers import Layer


class Sampling(Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), dtype=z_mean.dtype)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
