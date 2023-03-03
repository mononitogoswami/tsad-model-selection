import numpy as np
import torch as t
from re import L
import matplotlib.pyplot as plt

def de_unfold(windows, window_step):
    """
    windows of shape (n_windows, n_channels, window_size)
    """
    n_windows, n_channels, window_size = windows.shape

    if window_step < 0:
        window_step = window_size

    assert window_step <= window_size, 'Window step must be smaller than window_size'

    total_len = (n_windows)*window_step + (window_size-window_step)

    x = t.zeros((n_channels, total_len))
    counter = t.zeros((1, total_len))

    for i in range(n_windows):
        start = i*window_step
        end = start + window_size
        x[:, start:end] += windows[i]
        counter[:, start:end] += 1

    x = x/counter

    return x

def visualize_predictions(predictions: dict, savefig=True):
    """Visualizes univariate models given the predictions dictionary
    """
    MODEL_NAMES = list(predictions.keys())
    fig, axes = plt.subplots(len(MODEL_NAMES),
                             1,
                             sharey=True,
                             sharex=True,
                             figsize=(30, 5 * len(MODEL_NAMES)))

    for i, model_name in enumerate(MODEL_NAMES):
        start_anomaly = np.argmax(
            np.diff(predictions[model_name]['anomaly_labels'].flatten()))
        end_anomaly = np.argmin(
            np.diff(predictions[model_name]['anomaly_labels'].flatten()))
        axes[i].plot(predictions[model_name]['Y'].flatten(),
                     color='darkblue',
                     label='Y')
        axes[i].plot(predictions[model_name]['Y_hat'].flatten(),
                     color='darkgreen',
                     label='Y_hat')
        axes[i].plot(
            np.arange(start_anomaly, end_anomaly),
            predictions[model_name]['Y'].flatten()[start_anomaly:end_anomaly],
            color='red',
            label='Anomaly')

        entity_scores = predictions[model_name]['entity_scores'].flatten()
        entity_scores = (entity_scores - entity_scores.min()) / (
            entity_scores.max() - entity_scores.min())
        # entity_scores = (entity_scores - entity_scores.mean())/(entity_scores.std())
        axes[i].plot(entity_scores,
                     color='magenta',
                     linestyle='--',
                     label='Anomaly Scores')

        axes[i].set_title(f'Predictions of Model {model_name}', fontsize=16)
        axes[i].legend(fontsize=16, ncol=2, shadow=True, fancybox=True)
        axes[i].set_xlabel('Time', fontsize=16)
        axes[i].set_ylabel('Y', fontsize=16)

    if savefig:
        plt.savefig('predictions.pdf')
    plt.show()


def visualize_data(train_data, test_data, savefig=False):
    """Visualizes train and testing splits of a univariate entity.
    """
    # Visualize the train and the test data
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(25, 4))
    axes[0].plot(train_data.entities[0].Y.flatten(), color='darkblue')
    axes[0].set_title('Train data', fontsize=16)

    start_anomaly = np.argmax(np.diff(test_data.entities[0].labels.flatten()))
    end_anomaly = np.argmin(np.diff(test_data.entities[0].labels.flatten()))

    axes[1].plot(test_data.entities[0].Y.flatten(),
                 color='darkblue',
                 label='Y')
    axes[1].plot(np.arange(start_anomaly, end_anomaly),
                 test_data.entities[0].Y.flatten()[start_anomaly:end_anomaly],
                 color='red',
                 label='Anomaly')
    axes[1].set_title('Test data', fontsize=16)
    axes[1].legend(fontsize=16, ncol=2, shadow=True, fancybox=True)

    if savefig:
        plt.savefig('data_visual.pdf')
    plt.show()