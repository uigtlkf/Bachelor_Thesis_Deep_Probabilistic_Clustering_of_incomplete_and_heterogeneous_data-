import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from src.sampling import Sampling
from src.PoE.combineExpersLayer_PoE import CombineExpertsLayer_PoE


def encoder_expert(inputs, latent_dim, expert_id):
    x = Flatten()(inputs)
    x = Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    z_mean = Dense(latent_dim, name=f'z_mean_{expert_id}')(x)
    z_log_var = Dense(latent_dim, name=f'z_log_var_{expert_id}')(x)
    return z_mean, z_log_var

def vae_poe(input_shape, latent_dim, num_experts):
    inputs_list = [Input(shape=input_shape, name=f'input_{i}') for i in range(num_experts)]

    expert_outputs = []
    for i in range(num_experts):
        z_mean, z_log_var = encoder_expert(inputs_list[i], latent_dim, i)
        expert_outputs.append((z_mean, z_log_var))

    combine_layer = CombineExpertsLayer_PoE(num_experts)
    combined_mean, combined_log_var = combine_layer(expert_outputs)

    sampling_layer = Sampling()
    z = sampling_layer([combined_mean, combined_log_var])

    decoder_output = build_decoder(latent_dim)(z)

    model = Model(inputs=inputs_list, outputs=[decoder_output, combined_mean, combined_log_var])
    return model

def build_decoder(latent_dim):
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(4 * 7 * 1024, activation='relu')(latent_inputs)
    x = tf.keras.layers.Reshape((4, 7, 1024))(x)

    x = tf.keras.layers.Conv2DTranspose(1024, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    shortcut = x
    x = tf.keras.layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    shortcut = tf.keras.layers.Conv2DTranspose(256, (1, 1), activation='relu', padding='same')(shortcut)
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    attention = tf.keras.layers.Conv2DTranspose(256, (1, 1), activation='sigmoid')(x)
    x = tf.keras.layers.multiply([x, attention])

    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(28, (3, 3), activation='sigmoid', padding='same')(x)
    outputs = tf.keras.layers.Reshape((4, 7, 28))(x)
    decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    return decoder


def build_encoder(input_shape, latent_dim=2000, num_experts=4):
    inputs_list = [Input(shape=input_shape, name=f'input_{i}') for i in range(num_experts)]

    expert_outputs = []
    for i in range(num_experts):
        x = Flatten()(inputs_list[i])
        x = Dense(512)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        z_mean = Dense(latent_dim, name=f'z_mean_{i}')(x)
        z_log_var = Dense(latent_dim, name=f'z_log_var_{i}')(x)
        expert_outputs.append((z_mean, z_log_var))

    combine_layer = CombineExpertsLayer_PoE(num_experts)
    combined_mean, combined_log_var = combine_layer(expert_outputs)

    sampling_layer = Sampling()
    z = sampling_layer([combined_mean, combined_log_var])

    encoder = Model(inputs=inputs_list, outputs=[combined_mean, combined_log_var, z], name='encoder')
    encoder.summary()
    return encoder

