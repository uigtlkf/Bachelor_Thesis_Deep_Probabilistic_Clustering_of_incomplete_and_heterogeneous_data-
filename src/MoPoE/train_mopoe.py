import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from src.MoPoE.mopoe_vae import build_encoder, build_decoder
from src.missingness_evaluation import evaluate_on_missing_data, generate_missing_data

epsilon = 1e-6

def vae_loss(inputs, outputs, mean, log_var, kl_weight):
    reconstruction_loss = 0

    outputs = tf.reshape(outputs, [outputs.shape[0], 4, 7, 28, 1])

    for i in range(len(inputs)):
        input_part = inputs[i]
        output_part = outputs[:, i, :, :, :]
        assert input_part.shape == output_part.shape, f"Input part shape: {input_part.shape}, Output part shape: {output_part.shape}"
        reconstruction_loss += tf.reduce_mean(tf.square(input_part - output_part))

    kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=-1)
    total_loss = reconstruction_loss + kl_weight * tf.reduce_mean(kl_loss)

    return total_loss, reconstruction_loss, kl_loss


def cut_into_parts(images):
    parts = np.split(images, 4, axis=1)
    return np.stack(parts, axis=1)

def train_mopoe(missing_rate, mechanism, pattern, seed, epochs):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    latent_dim = 2000
    batch_size = 128
    (train_images, _), (test_images, true_labels) = tf.keras.datasets.mnist.load_data()

    train_images = generate_missing_data(train_images, missing_rate, mechanism, pattern)
    test_images = generate_missing_data(test_images, missing_rate, mechanism, pattern)

    train_images = train_images.astype('float32') / 255.
    test_images = test_images.astype('float32') / 255.

    train_images_parts = cut_into_parts(train_images)
    test_images_parts = cut_into_parts(test_images)

    train_images_parts = np.nan_to_num(train_images_parts, nan=0.0)
    test_images_parts = np.nan_to_num(test_images_parts, nan=0.0)

    K = tf.keras.backend
    kl_weight = K.variable(0.0)
    kl_weight_increase = 0.05

    encoder = build_encoder(latent_dim, num_experts=4)
    decoder = build_decoder(latent_dim)

    inputs_list = [Input(shape=(7, 28, 1), name=f'input_{i}') for i in range(4)]

    mean, log_var, z = encoder(inputs_list)

    z_dense = Dense(latent_dim)(z)

    reconstructed_output = decoder(z_dense)

    vae_model_vanilla = Model(inputs_list, [reconstructed_output, mean, log_var], name='vae_mopoe')
    optimizer_mopoe = Adam(learning_rate=0.0001)
    vae_model_vanilla.compile(optimizer=optimizer_mopoe, loss=lambda x, y: vae_loss(x, y, mean, log_var, kl_weight))

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images_parts)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

    for epoch in range(epochs):
        total_loss, total_recon_loss, total_kl_loss = 0, 0, 0
        for batch in train_dataset:
            batch_split = [tf.expand_dims(batch[:, i, :, :], axis=-1) for i in range(4)]

            with tf.GradientTape() as tape:
                reconstructed_output, mean, log_var = vae_model_vanilla(batch_split)
                loss, recon_loss, kl_loss = vae_loss(batch_split, reconstructed_output, mean, log_var, kl_weight)
            gradients = tape.gradient(loss, vae_model_vanilla.trainable_variables)
            optimizer_mopoe.apply_gradients(zip(gradients, vae_model_vanilla.trainable_variables))

            total_loss += loss.numpy()
            total_recon_loss += recon_loss.numpy()
            total_kl_loss += kl_loss.numpy()

        if kl_weight < 1.0:
            K.set_value(kl_weight, K.get_value(kl_weight) + kl_weight_increase)

        print(
            f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataset)}, Recon Loss: {total_recon_loss / len(train_dataset)}, KL Loss: {total_kl_loss / len(train_dataset)}")

    test_dataset = tf.data.Dataset.from_tensor_slices(test_images_parts).batch(batch_size)
    reconstructed_images_mopoe = []
    for batch in test_dataset:
        batch_split = [tf.expand_dims(batch[:, i, :, :], axis=-1) for i in range(4)]
        test_output_mopoe = vae_model_vanilla.predict(batch_split)
        reconstructed_images_mopoe.append(test_output_mopoe[0])

    reconstructed_images_mopoe = np.concatenate(reconstructed_images_mopoe, axis=0).reshape(-1, 4, 7, 28)

    reconstructed_images_mopoe_flattened = reconstructed_images_mopoe.reshape(reconstructed_images_mopoe.shape[0], -1)

    result = evaluate_on_missing_data(true_labels, reconstructed_images_mopoe_flattened, missing_rate, mechanism,
                                            pattern,seed, n_clusters=10, type='mopoe')

    return test_images, reconstructed_images_mopoe, true_labels, result
