import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class CombineExpertsLayer_MoE(Layer):
    def __init__(self, num_experts, latent_dim, **kwargs):
        super(CombineExpertsLayer_MoE, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.latent_dim = latent_dim
        self.gating_network = Dense(num_experts, activation='softmax')

    def call(self, inputs):
        gating_input = inputs[0]
        experts_outputs = inputs[1:]

        if isinstance(gating_input, (list, tuple)):
            gating_input = tf.concat(gating_input, axis=-1)

        gating_weights = self.gating_network(gating_input)
        print(f"Gating weights after network: {gating_weights.shape}")

        if len(gating_weights.shape) == 3:
            gating_weights = tf.reduce_mean(gating_weights, axis=1)
        gating_weights = tf.expand_dims(gating_weights, axis=-1)
        gating_weights = tf.expand_dims(gating_weights, axis=-1)
        print(f"Gating weights after reshape: {gating_weights.shape}")

        combined_mean = 0
        combined_log_var = 0


        for i, expert_output in enumerate(experts_outputs):
            mean, log_var = expert_output

            if len(mean.shape) == 2:
                mean = tf.expand_dims(mean, axis=1)
                log_var = tf.expand_dims(log_var, axis=1)

            combined_mean += gating_weights[:, i, :, :] * mean
            combined_log_var += gating_weights[:, i, :, :] * log_var

        combined_mean = tf.squeeze(combined_mean, axis=1)
        combined_log_var = tf.squeeze(combined_log_var, axis=1)

        return combined_mean, combined_log_var
