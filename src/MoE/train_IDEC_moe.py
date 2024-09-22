from sklearn.cluster import KMeans
from src.IDEC import *
import matplotlib.pyplot as plt
from src.missingness_evaluation import evaluate_on_missing_data, compute_accuracy, generate_missing_data
from src.MoE.train_moe import train_moe, safe_savefig
import numpy as np
from src.MoE.moe_vae import *

def compute_custom_loss(reconstructed_images, targets):
    reconstructed_images = tf.reshape(reconstructed_images, tf.shape(targets))
    mse_loss = tf.reduce_mean(tf.square(reconstructed_images - targets))
    bce_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(targets, reconstructed_images))
    return mse_loss + bce_loss

def learning_rate_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def compute_kl_loss(mean, log_var):
    kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
    return tf.reduce_mean(kl_loss)


def kl_annealing(epoch, total_epochs):
    return min(1.0, epoch / total_epochs)


def learning_rate_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
def cut_into_parts(images):
    parts = np.split(images, 4, axis=1)
    return np.stack(parts, axis=1)

def train_idec(train_images, true_labels, latent_dim, n_clusters, epochs, missing_rate, mechanism, batch_size, pattern,seed, save_freq=5):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    num_experts = 4
    input_shape = (7, 28)

    encoder = build_encoder(input_shape, latent_dim, num_experts)
    decoder = build_decoder(latent_dim)
    idec_model = IDEC(encoder=encoder, decoder=decoder, n_clusters=n_clusters, latent_dim=latent_dim)

    initial_lr = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr, clipvalue=1.0)
    train_images = generate_missing_data(train_images, missing_rate, mechanism, pattern)
    train_images_parts = cut_into_parts(train_images)

    train_images_parts = np.nan_to_num(train_images_parts, nan=0.0)

    dataset = tf.data.Dataset.from_tensor_slices(train_images_parts).batch(batch_size)

    best_loss = np.inf
    patience = 10
    patience_counter = 0
    checkpoint_filepath = 'best_idec_model.h5'

    history = {'loss': [], 'kl_loss': [], 'reconstruction_loss': [], 'accuracy': []}

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_reconstruction_loss = 0.0
        step_count = 0

        for step, batch_images in enumerate(dataset):
            batch_size_actual = batch_images.shape[0]
            print(f"Batch {step}, Batch Shape: {batch_images.shape}")

            inputs_list = [tf.squeeze(batch_images[:, i, :, :, :], axis=1) for i in range(num_experts)]

            with tf.GradientTape() as tape:
                for i, input_tensor in enumerate(inputs_list):
                    print(f"Input {i} shape: {input_tensor.shape}")

                z_mean, z_log_var, z = idec_model.encoder(inputs_list)
                reconstructed_images = idec_model.decoder(z)

                reconstructed_images = tf.reshape(reconstructed_images, batch_images.shape)

                reconstruction_loss = compute_custom_loss(reconstructed_images, batch_images)
                kl_loss = compute_kl_loss(z_mean, z_log_var)
                loss = reconstruction_loss + kl_annealing(epoch, epochs) * kl_loss

            gradients = tape.gradient(loss, idec_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, idec_model.trainable_variables))

            epoch_loss += loss.numpy() * batch_size_actual
            epoch_kl_loss += kl_loss.numpy() * batch_size_actual
            epoch_reconstruction_loss += reconstruction_loss.numpy() * batch_size_actual
            step_count += batch_size_actual

            if step % 100 == 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.numpy()}, KL Loss: {kl_loss.numpy()}, Reconstruction Loss: {reconstruction_loss.numpy()}")

        epoch_loss /= step_count
        epoch_kl_loss /= step_count
        epoch_reconstruction_loss /= step_count
        history['loss'].append(epoch_loss)
        history['kl_loss'].append(epoch_kl_loss)
        history['reconstruction_loss'].append(epoch_reconstruction_loss)

        train_images_list = [tf.squeeze(train_images_parts[:, i, :, :, :], axis=1) for i in range(num_experts)]
        z_mean, _, _ = idec_model.encoder.predict(train_images_list)
        kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(z_mean)
        predicted_labels = kmeans.predict(z_mean)
        accuracy = compute_accuracy(true_labels, predicted_labels, n_clusters)
        history['accuracy'].append(accuracy)

        print(f"Epoch {epoch + 1}, Average Epoch Loss: {epoch_loss}, Average KL Loss: {epoch_kl_loss}, Average Reconstruction Loss: {epoch_reconstruction_loss}, Accuracy: {accuracy}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            if (epoch + 1) % save_freq == 0:
                idec_model.save_weights(checkpoint_filepath)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                idec_model.load_weights(checkpoint_filepath)
                break

        new_lr = learning_rate_schedule(epoch, initial_lr)
        tf.keras.backend.set_value(optimizer.learning_rate, new_lr)

    return idec_model, history


def print_images_idec(original_images, reconstructed_images, new_reconstructions, missing_rate, mechanism, pattern,seed, num_images=10):
    fig, axes = plt.subplots(3, num_images, figsize=(20, 4))
    for i in range(num_images):
        for part in range(4):
            ax = axes[0, i]
            ax.imshow(original_images[i, part].squeeze(), cmap='gray')
            ax.axis('off')

            ax = axes[1, i]
            ax.imshow(reconstructed_images[i, part].squeeze(), cmap='gray')
            ax.axis('off')

            ax = axes[2, i]
            ax.imshow(new_reconstructions[i, part].squeeze(), cmap='gray')
            ax.axis('off')

    plt.tight_layout()
    safe_savefig(f'moe_IDEC_results_comparison_{missing_rate}_{mechanism}_{pattern}_{seed}.png')
    plt.show()
    plt.close(fig)
    print("END")
def summarize_results(results, mechanism, pattern):
    metrics = ['nmi', 'accuracy', 'std_error']
    missing_rates = sorted(list(set(result['missing_rate'] for result in results)))
    seeds = sorted(list(set(result['seed'] for result in results)))

    results_moe = [result for result in results if result['type'] == 'moe' and result['mechanism'] == mechanism and result['pattern'] == pattern]
    results_idec = [result for result in results if result['type'] == 'idec' and result['mechanism'] == mechanism and result['pattern'] == pattern]

    def plot_results_for_seed(results_moe, results_idec, seed):
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 18))

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            data_poe = []
            for rate in missing_rates:
                data_poe.append(
                    np.mean([result[metric] for result in results_moe if
                             result['seed'] == seed and result['missing_rate'] == rate]))

            data_idec = []
            for rate in missing_rates:
                data_idec.append(
                    np.mean([result[metric] for result in results_idec if
                             result['seed'] == seed and result['missing_rate'] == rate]))

            if all(d is not None for d in data_poe):
                ax.plot(missing_rates, data_poe, marker='o', linestyle='-',
                        label=f'MoE {mechanism} pattern {pattern} - Seed {seed}')

            if all(d is not None for d in data_idec):
                ax.plot(missing_rates, data_idec, marker='^', linestyle='--',
                        label=f'IDEC {mechanism} pattern {pattern} - Seed {seed}')

            ax.set_xticks(missing_rates)
            ax.set_xticklabels(missing_rates)
            ax.set_title(f'{metric.upper()} Comparison for Seed {seed}')
            ax.set_xlabel('Missing Rate')
            ax.set_ylabel(metric.upper())
            ax.legend(title='Algorithm')

        plt.tight_layout()
        safe_savefig(f'MoE_IDEC_{mechanism}_pattern_{pattern}_seed_{seed}_comparison.png')
        plt.clf()
        plt.close(fig)

    for seed in seeds:
        plot_results_for_seed(results_moe, results_idec, seed)


if __name__ == "__main__":
    missing_rates = [0.1, 0.3, 0.5]
    mechanisms = ['MCAR', 'MAR', 'MNAR']
    patterns = [1, 2]
    n_clusters = 10
    epochs = 1
    batch_size = 128
    latent_dim = 2000
    seeds = [7, 42, 123]
    for mechanism in mechanisms:
        for pattern in patterns:
            results = []
            for missing_rate in missing_rates:
                for seed in seeds:
                    original_images, reconstructed_images, true_labels, result_moe = train_moe(missing_rate, mechanism, pattern, seed, epochs)
                    result_moe = evaluate_on_missing_data(true_labels, reconstructed_images, missing_rate, mechanism,
                                                              pattern, seed, n_clusters, type='moe')

                    idec_model, history = train_idec(original_images, true_labels, latent_dim, n_clusters, epochs, missing_rate, mechanism, batch_size, pattern,seed)

                    train_images_parts = np.reshape(original_images, (-1, 4, 7, 28, 1))

                    inputs_list = [train_images_parts[:, i, :, :, :] for i in range(4)]

                    z_mean, _, _ = idec_model.encoder.predict(inputs_list)

                    latent_representations = idec_model.predict(inputs_list)
                    latent_representations = latent_representations[0].reshape(-1, 4, 7, 28, 1)

                    result_idec_moe = evaluate_on_missing_data(true_labels, z_mean, missing_rate, mechanism, pattern,seed, n_clusters, type='idec')

                    print_images_idec(original_images, reconstructed_images, latent_representations, missing_rate, mechanism, pattern,seed, num_images=10)

                    result_moe['pattern'] = pattern
                    result_idec_moe['pattern'] = pattern

                    result_moe['seed'] = seed
                    result_idec_moe['seed'] = seed

                    results.append(result_moe)
                    results.append(result_idec_moe)

            summarize_results(results, mechanism, pattern)
