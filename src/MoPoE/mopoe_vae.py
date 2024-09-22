from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, BatchNormalization, Dropout, add, multiply, GaussianNoise, Concatenate, LeakyReLU
from tensorflow.keras.models import Model
from src.sampling import *
import tensorflow as tf

epsilon = 1e-6
def build_expert(inputs, latent_dim):
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.4)(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    return z_mean, z_log_var

def combine_experts_poe(experts_means, experts_log_vars):
    precision_weights = [tf.exp(-log_var) for log_var in experts_log_vars]
    combined_mean = tf.add_n([mean * weight for mean, weight in zip(experts_means, precision_weights)]) / tf.add_n(precision_weights)
    combined_log_var = -tf.math.log(tf.add_n(precision_weights) + epsilon)
    return combined_mean, combined_log_var


def combine_experts_moe(poe_means, poe_log_vars, gating_weights):
    poe_means = tf.stack(poe_means, axis=1)
    poe_log_vars = tf.stack(poe_log_vars, axis=1)

    gating_weights = tf.expand_dims(gating_weights, axis=-1)
    weighted_means = tf.reduce_sum(gating_weights * poe_means, axis=1)
    weighted_log_vars = tf.reduce_sum(gating_weights * tf.exp(poe_log_vars), axis=1)
    combined_log_var = tf.math.log(weighted_log_vars + epsilon)

    return weighted_means, combined_log_var


def build_encoder(latent_dim, num_experts=4, num_poes=3):
    inputs_list = [Input(shape=(7, 28, 1), name=f'input_{i}') for i in range(num_experts)]

    experts_means_log_vars = [build_expert(inputs, latent_dim) for inputs in inputs_list]
    experts_means, experts_log_vars = zip(*experts_means_log_vars)

    poe_means_list = []
    poe_log_vars_list = []

    for _ in range(num_poes):
        combined_mean_poe, combined_log_var_poe = combine_experts_poe(experts_means, experts_log_vars)
        poe_means_list.append(combined_mean_poe)
        poe_log_vars_list.append(combined_log_var_poe)

    flattened_inputs = [Flatten()(inputs) for inputs in inputs_list]
    gating_input = Concatenate()(flattened_inputs)
    gating_weights = Dense(num_poes, activation='softmax')(gating_input)

    combined_mean, combined_log_var = combine_experts_moe(poe_means_list, poe_log_vars_list, gating_weights)

    z = Sampling()([combined_mean, combined_log_var])

    encoder = Model(inputs_list, [combined_mean, combined_log_var, z], name='encoder')
    encoder.summary()
    return encoder


def build_decoder(latent_dim):
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(4096, activation='relu')(latent_inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(4 * 7 * 1024, activation='relu')(x)
    x = Reshape((4, 7, 1024))(x)
    x = Conv2DTranspose(1024, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    shortcut = x
    x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    shortcut = Conv2DTranspose(256, (1, 1), activation='relu', padding='same')(shortcut)
    x = add([x, shortcut])
    x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    shortcut = x
    x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    shortcut = Conv2DTranspose(256, (1, 1), activation='relu', padding='same')(shortcut)
    x = add([x, shortcut])
    x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    attention = Conv2DTranspose(256, (1, 1), activation='sigmoid')(x)
    x = multiply([x, attention])
    x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(28, (3, 3), activation='sigmoid', padding='same')(x)
    outputs = Reshape((4, 7, 28))(x)
    decoder = Model(latent_inputs, outputs, name='decoder')
    return decoder
