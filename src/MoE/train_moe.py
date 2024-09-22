from src.missingness_evaluation import evaluate_on_missing_data,generate_missing_data
from src.MoE.moe_vae import vae_moe
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

def check_disk_space(path='/mnt/data'):
    st = os.statvfs(path)
    free_space = st.f_bavail * st.f_frsize
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

def train_moe(missing_rate, mechanism, pattern, seed, epochs, batch_size=128):
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

    input_shape = (7, 28)
    latent_dim = 2000
    num_experts = 4

    vae_model_vanilla = vae_moe([input_shape] * num_experts, latent_dim, num_experts)
    optimizer_moe = tf.keras.optimizers.Adam(learning_rate=0.0001)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_dataset:
            inputs_list = [batch[:, i, :, :] for i in range(num_experts)]

            with tf.GradientTape() as tape:
                for i, input_tensor in enumerate(inputs_list):
                    print(f"Input {i} shape: {input_tensor.shape}")

                reconstructed_output, mean, log_var = vae_model_vanilla(inputs_list)
                loss, recon_loss, kl_loss = vae_loss(batch, reconstructed_output, mean, log_var, kl_weight)

            gradients = tape.gradient(loss, vae_model_vanilla.trainable_variables)
            optimizer_moe.apply_gradients(zip(gradients, vae_model_vanilla.trainable_variables))

            total_loss += loss.numpy()

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

    test_inputs_list = [test_images[:, i, :, :] for i in range(num_experts)]
    test_output_moe = vae_model_vanilla(test_inputs_list)
    reconstructed_images_moe = test_output_moe[0].numpy().reshape(-1, 4, 7, 28)

    result = evaluate_on_missing_data(true_labels, reconstructed_images_moe, missing_rate, mechanism, pattern,seed, n_clusters=10, type='moe')
    print_images(test_images, reconstructed_images_moe, missing_rate, mechanism, pattern,seed)

    return test_images, reconstructed_images_moe, true_labels, result
def print_images(test_images, reconstructed_images_moe, missing_rate, mechanism, pattern,seed):
    n = 10
    fig, axes = plt.subplots(2, n, figsize=(20, 4))
    for i in range(n):
        for part in range(4):
            ax = axes[0, i]
            ax.imshow(test_images[i, part].squeeze(), cmap='gray')
            ax.set_title("Original Part {}".format(part + 1))
            ax.axis('off')

            ax = axes[1, i]
            ax.imshow(reconstructed_images_moe[i, part].squeeze(), cmap='gray')
            ax.set_title("Reconstructed Part {}".format(part + 1))
            ax.axis('off')

    plt.tight_layout()
    safe_savefig(f'moe_results_{missing_rate}_{mechanism}_{pattern}_{seed}.png')
    plt.show()
    plt.close(fig)