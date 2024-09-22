import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import sem
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

def generate_missing_data(data, missing_rate, mechanism, pattern):
    missing_data = data.astype('float32').copy()
    num_elements = np.prod(data.shape)
    num_missing = int(missing_rate * num_elements)

    if mechanism == 'MCAR':
        if pattern == 1:
            # Pattern 1: Random selection
            missing_indices = np.random.choice(num_elements, num_missing, replace=False)
        elif pattern == 2:
            # Pattern 2: Block-wise missingness
            indices = np.arange(num_elements)
            np.random.shuffle(indices)
            missing_indices = indices[:num_missing]
    elif mechanism == 'MAR':
        if pattern == 1:
            # Pattern 1: Sorted based on values
            sorted_indices = np.argsort(data.flatten())
            missing_indices = sorted_indices[:num_missing]
        elif pattern == 2:
            # Pattern 2: Conditional on threshold of one variable, distributed missingness
            threshold = np.percentile(data, 50)
            condition_indices = np.where(data.flatten() > threshold)[0]
            np.random.shuffle(condition_indices)
            if len(condition_indices) < num_missing:
                missing_indices = np.random.choice(condition_indices, num_missing, replace=True)
            else:
                missing_indices = condition_indices[:num_missing]
    elif mechanism == 'MNAR':
        if pattern == 1:
            # Pattern 1: Sorted in descending order
            sorted_indices = np.argsort(-data.flatten())
            missing_indices = sorted_indices[:num_missing]
        elif pattern == 2:
            # Pattern 2: Conditional on sum of pairs of variables
            data_sum = data.flatten() + np.roll(data.flatten(), shift=1)
            sorted_indices = np.argsort(-data_sum)
            missing_indices = sorted_indices[:num_missing]
    else:
        raise ValueError("Unsupported mechanism: choose from 'MCAR', 'MAR', 'MNAR'")

    missing_data.flat[missing_indices] = np.nan
    return missing_data
def evaluate_performance(true_labels, predicted_labels):
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    error = np.abs(true_labels - predicted_labels)
    std_error = sem(error)
    return nmi, std_error

def compute_accuracy(true_labels, predicted_labels, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int32)
    for i in range(n_clusters):
        for j in range(n_clusters):
            cost_matrix[i, j] = np.sum((true_labels == i) & (predicted_labels == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    accuracy = cost_matrix[row_ind, col_ind].sum() / true_labels.size
    return accuracy

def evaluate_on_missing_data(true_labels, latent_representations, missing_rate, mechanism, pattern, seed, n_clusters, type):
    if(type!='idec'):
        latent_representations = latent_representations.reshape(latent_representations.shape[0], -1)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(latent_representations)
    predicted_labels = kmeans.predict(latent_representations)

    if len(predicted_labels) != len(true_labels):
        raise ValueError(f"Length mismatch: {len(predicted_labels)} (predicted) vs {true_labels.shape[0]} (true)")

    accuracy = compute_accuracy(true_labels, predicted_labels, n_clusters)

    nmi, se = evaluate_performance(true_labels, predicted_labels)

    result = {
        'missing_rate': missing_rate,
        'mechanism': mechanism,
        'nmi': nmi,
        'accuracy': accuracy,
        'std_error': se,
        'pattern': pattern,
        'type': type,
        'seed':seed
    }

    print("Result:")
    print(result)

    metrics = ['nmi', 'accuracy', 'std_error']
    data = {metric: [result[metric]] for metric in metrics}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x = np.arange(len([missing_rate]))
    width = 0.25

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.bar(x, data[metric], width, label=mechanism)

        ax.set_xticks(x)
        ax.set_xticklabels([missing_rate])
        ax.set_title(f'{metric.upper()} for missingness rate {missing_rate} and mechanism {mechanism}')
        ax.set_xlabel('Missing Rate')
        ax.set_ylabel(metric.upper())
        ax.legend(title='Mechanism')

    plt.tight_layout()
    return result
