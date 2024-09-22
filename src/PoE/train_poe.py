import tensorflow as tf
import matplotlib.pyplot as plt
from src.PoE.poe_vae import vae_poe
import numpy as np
import os
from src.missingness_evaluation import evaluate_on_missing_data, generate_missing_data

def vae_loss(inputs, outputs, mean, log_var, kl_weight):
    total_loss = 0
    num_parts = inputs.shape[1]
    outputs = tf.reshape(outputs, inputs.shape)
    for i in range(num_parts):
        input_part = inputs[:, i, :, :]
        output_part = outputs[:, i, :, :]
        assert input_part.shape == output_part.shape, f"Input part shape: {input_part.shape}, Output part shape: {output_part.shape}"
        reconstruction_loss = tf.reduce_mean(tf.square(input_part - output_part))
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=-1)
        total_loss += reconstruction_loss + kl_weight * tf.reduce_mean(kl_loss)
    return total_loss, reconstruction_loss, kl_loss


def cut_into_parts(images):
    parts = np.split(images, 4, axis=1)
    return np.stack(parts, axis=1)

def train_poe(missing_rate, mechanism, pattern, seed, epochs):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    (train_images, _), (test_images, true_labels) = tf.keras.datasets.mnist.load_data()

    train_images = generate_missing_data(train_images, missing_rate, mechanism, pattern)
    test_images = generate_missing_data(test_images, missing_rate, mechanism, pattern)

    train_images = train_images.astype('float32') / 255.
    test_images = test_images.astype('float32') / 255.

    train_images = cut_into_parts(train_images)
    test_images = cut_into_parts(test_images)

    train_images = np.nan_to_num(train_images, nan=0.0)
    test_images = np.nan_to_num(test_images, nan=0.0)

    K = tf.keras.backend
    kl_weight = K.variable(0.0)
    kl_weight_increase = 0.05

    input_shape = (7, 28)
    latent_dim = 2000
    num_experts = 4

    vae_model_vanilla = vae_poe(input_shape, latent_dim, num_experts)
    optimizer_poe = tf.keras.optimizers.Adam(learning_rate=0.0001)

    batch_size = 128

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

    for epoch in range(epochs):
        total_loss, total_recon_loss, total_kl_loss = 0, 0, 0
        for batch in train_dataset:
            inputs_list = [batch[:, i, :, :] for i in range(num_experts)]

            with tf.GradientTape() as tape:
                reconstructed_output, mean, log_var = vae_model_vanilla(inputs_list)
                print(f"Original batch shape: {batch.shape}, Reconstructed output shape: {reconstructed_output.shape}")
                loss, recon_loss, kl_loss = vae_loss(batch, reconstructed_output, mean, log_var, kl_weight)
            gradients = tape.gradient(loss, vae_model_vanilla.trainable_variables)
            optimizer_poe.apply_gradients(zip(gradients, vae_model_vanilla.trainable_variables))

            total_loss += loss.numpy()
            total_recon_loss += recon_loss.numpy()
            total_kl_loss += kl_loss.numpy()

        if kl_weight < 1.0:
            K.set_value(kl_weight, K.get_value(kl_weight) + kl_weight_increase)

        print(
            f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataset)}, Recon Loss: {total_recon_loss / len(train_dataset)}, KL Loss: {total_kl_loss / len(train_dataset)}")

    test_inputs_list = [test_images[:, i, :, :] for i in range(num_experts)]
    test_output_poe = vae_model_vanilla(test_inputs_list)
    reconstructed_images_poe = test_output_poe[0].numpy().reshape(-1, 4, 7, 28)

    reconstructed_images_poe_flattened = reconstructed_images_poe.reshape(reconstructed_images_poe.shape[0], -1)

    result = evaluate_on_missing_data(true_labels, reconstructed_images_poe_flattened, missing_rate, mechanism,
                                          pattern, seed,n_clusters=10, type='poe')
    print_images(test_images, reconstructed_images_poe_flattened, missing_rate, mechanism, pattern,seed)

    return test_images, reconstructed_images_poe, true_labels, result

def print_images(test_images, reconstructed_images_poe_flattened, missing_rate, mechanism, pattern,seed,num_images=10):
    fig, axes = plt.subplots(2, num_images, figsize=(20, 4))
    for i in range(num_images):
        for part in range(4):
            ax = axes[0, i]
            ax.imshow(test_images[i, part], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"Original Part {part + 1}")
            ax.axis('off')

            reconstructed_image = reconstructed_images_poe_flattened[i].reshape(4, 7, 28)
            ax = axes[1, i]
            ax.imshow(reconstructed_image[part], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"Reconstructed Part {part + 1}")
            ax.axis('off')

    plt.tight_layout()
    safe_savefig(f'poe_results_{missing_rate}_{mechanism}_{pattern}_{seed}.png', dpi=300)
    plt.close(fig)
def check_disk_space(path='/'):
    statvfs = os.statvfs(path)
    free_space = statvfs.f_frsize * statvfs.f_bavail
    return free_space

def safe_savefig(filename, min_free_space=100 * 1024 * 1024, dpi=100, format='png'):
    try:
        free_space = check_disk_space()
        if free_space > min_free_space:
            plt.savefig(filename, dpi=dpi, format=format, bbox_inches='tight', quality=95)
            print(f"Saved figure: {filename}")
        else:
            print(f"Not enough disk space to save the figure: {filename}")
    except Exception as e:
        print(f"Error saving figure {filename}: {e}")

