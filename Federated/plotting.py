import sys
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt

def plot_accuracy_vs_rounds(round_accuracies, training_rounds):
    """
    Plot model accuracy vs rounds.

    Args:
        round_accuracies (list): List of accuracy values per round.
        training_rounds (int): Number of training rounds.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, training_rounds + 1), round_accuracies, 'b-o')
    ax.set_title('Model Accuracy vs Rounds')
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_client_training_times(client_times, avg_round_times, training_rounds):
    """
    Plot client training times vs rounds.

    Args:
        client_times (list): List of training times for each client.
        avg_round_times (list): Average training times per round.
        training_rounds (int): Number of training rounds.
    """
    fig, ax = plt.subplots(figsize=(15, 5))

    rounds = range(1, training_rounds + 1)
    for i, times in enumerate(client_times):
        ax.plot(rounds, times, 'o--', alpha=0.5, label=f'Client {i+1}')
    ax.plot(rounds, avg_round_times, 'r-o', linewidth=2, label='Average')

    ax.set_title('Client Training Times per Round')
    ax.set_xlabel('Round')
    ax.set_ylabel('Time (seconds)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
