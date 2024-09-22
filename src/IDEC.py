from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense
import tensorflow.keras.backend as K
import tensorflow as tf


class IDEC(Model):
    def __init__(self, encoder, decoder, n_clusters, alpha=1.0, latent_dim=2000):
        super(IDEC, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.clustering_layer = Dense(n_clusters, activation='softmax')
        self.cluster_centers = K.variable(tf.zeros((n_clusters, latent_dim)), dtype=tf.float32)


    def call(self, inputs):
        encoder_output = self.encoder(inputs)
        z_mean, z_log_var, z = encoder_output
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        q = self.clustering_layer(z)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, q

    def compile(self, optimizer, loss_fn):
        super(IDEC, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            x_reconstructed, q = self(x, training=True)
            p = self.target_distribution(q)
            reconstruction_loss = self.loss_fn(x, x_reconstructed)
            kl_loss = tf.reduce_mean(tf.reduce_sum(p * tf.math.log(p / (q + 1e-10)), axis=1))
            loss = reconstruction_loss + self.alpha * kl_loss

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}

    def predict_clusters(self, inputs):
        z, _ = self(inputs)
        q = self.clustering_layer(z)
        return tf.argmax(q, axis=1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / tf.reduce_sum(q, axis=0)
        return weight / tf.reduce_sum(weight, axis=1, keepdims=True)
