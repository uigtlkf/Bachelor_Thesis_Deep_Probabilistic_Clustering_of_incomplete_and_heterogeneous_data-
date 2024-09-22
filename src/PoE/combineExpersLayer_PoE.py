from tensorflow.keras.layers import Layer, Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, BatchNormalization
import tensorflow as tf
class CombineExpertsLayer_PoE(Layer):
    def __init__(self, num_experts, **kwargs):
        super(CombineExpertsLayer_PoE, self).__init__(**kwargs)
        self.num_experts = num_experts

    def call(self, inputs):
        means = [inp[0] for inp in inputs]
        log_vars = [inp[1] for inp in inputs]

        precisions = [tf.exp(-log_var) for log_var in log_vars]
        precision_sum = tf.reduce_sum(precisions, axis=0)

        weighted_means = [mean * precision for mean, precision in zip(means, precisions)]
        weighted_mean_sum = tf.reduce_sum(weighted_means, axis=0)

        combined_mean = weighted_mean_sum / precision_sum
        combined_log_var = -tf.math.log(precision_sum)

        return combined_mean, combined_log_var