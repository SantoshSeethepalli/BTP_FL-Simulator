import matplotlib.pyplot as plt
import seaborn as sns

def plot_multi_scheme_comparison(results):
    """Plot comparison of multiple privacy schemes"""
    plt.style.use('seaborn')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = sns.color_palette("husl", len(results))
    
    # Accuracy over rounds
    for (scheme, data), color in zip(results.items(), colors):
        ax1.plot(data['fl_accs'], label=scheme, color=color)
    ax1.set_title('Federated Learning Accuracy by Scheme')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Training time per round
    for (scheme, data), color in zip(results.items(), colors):
        ax2.plot(data['fl_times'], label=scheme, color=color)
    ax2.set_title('Training Time per Round')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Time (s)')
    ax2.legend()
    
    # Final accuracy comparison
    schemes = list(results.keys())
    final_accs = [data['fl_accs'][-1] for data in results.values()]
    ax3.bar(schemes, final_accs)
    ax3.set_title('Final Accuracy Comparison')
    ax3.set_xticklabels(schemes, rotation=45)
    ax3.set_ylabel('Accuracy')
    
    # Total training time comparison
    total_times = [sum(data['fl_times']) for data in results.values()]
    ax4.bar(schemes, total_times)
    ax4.set_title('Total Training Time Comparison')
    ax4.set_xticklabels(schemes, rotation=45)
    ax4.set_ylabel('Time (s)')
    
    plt.tight_layout()
    plt.show()