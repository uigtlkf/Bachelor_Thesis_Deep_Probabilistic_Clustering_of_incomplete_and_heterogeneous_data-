from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from src.sampling import *
from src.MoE.combineExpersLayer_MoE import CombineExpertsLayer_MoE



def encoder_expert(inputs, latent_dim, expert_id):
    x = Flatten()(inputs)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    mean = Dense(latent_dim, name=f'z_mean_{expert_id}')(x)
    log_var = Dense(latent_dim, name=f'z_log_var_{expert_id}')(x)
    return mean, log_var


def vae_moe(input_shapes, latent_dim, num_experts=4):
    inputs_list = [Input(shape=input_shape, name=f'input_{i}') for i, input_shape in enumerate(input_shapes)]

    experts_outputs = [encoder_expert(inputs_list[i], latent_dim, i) for i in range(num_experts)]

    gating_input = tf.concat(inputs_list, axis=-1)

    combine_layer = CombineExpertsLayer_MoE(num_experts, latent_dim)
    combined_mean, combined_log_var = combine_layer([gating_input] + experts_outputs)

    sampling_layer = Sampling()
    z = sampling_layer([combined_mean, combined_log_var])

    decoder_model = decoder(latent_dim)
    reconstructed = decoder_model(z)

    return Model(inputs=inputs_list, outputs=[reconstructed, combined_mean, combined_log_var])




def build_encoder(input_shape, latent_dim=2000, num_experts=4):
    inputs_list = [Input(shape=input_shape, name=f'input_{i}') for i in range(num_experts)]

    experts_outputs = [encoder_expert(inputs_list[i], latent_dim, i) for i in range(num_experts)]

    combine_layer = CombineExpertsLayer_MoE(num_experts, latent_dim)
    combined_mean, combined_log_var = combine_layer(experts_outputs)

    sampling_layer = Sampling()
    z = sampling_layer([combined_mean, combined_log_var])

    encoder_model = Model(inputs=inputs_list, outputs=[combined_mean, combined_log_var, z], name='encoder')
    encoder_model.summary()
    return encoder_model



def build_decoder(latent_dim):
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(2048, activation='relu')(latent_inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(4 * 7 * 1024, activation='relu')(x)
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

def decoder(latent_dim):
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(4 * 7 * 256, activation='relu')(latent_inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Reshape((4, 7, 256))(x)

    x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(28, (3, 3), activation='sigmoid', padding='same')(x)
    outputs = Reshape((4, 7, 28))(x)
    return Model(latent_inputs, outputs, name='decoder')
