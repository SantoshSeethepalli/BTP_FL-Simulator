def show_comparison(fl_accs, fl_times, cent_accs, cent_times):
    """Plot Federated vs. Centralized accuracy and time using Tkinter GUI."""
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt

    # Create the Tkinter window
    root = tk.Tk()
    root.title("Federated vs. Centralized Training Comparison")

    # Prepare data
    rounds = list(range(1, len(fl_accs) + 1))

    # Create figure with improved layout
    fig, axs = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)

    # --- Accuracy Plot ---
    axs[0].plot(rounds, fl_accs, marker='o', label='Federated', color='tab:blue')
    axs[0].plot(rounds, cent_accs, marker='x', label='Centralized', color='tab:orange')
    axs[0].set_title("Accuracy per Round/Epoch", fontsize=12)
    axs[0].set_xlabel("Round/Epoch", fontsize=10)
    axs[0].set_ylabel("Accuracy", fontsize=10)
    axs[0].legend()
    axs[0].grid(True)

    # --- Time Plot ---
    axs[1].plot(rounds, fl_times, marker='o', label='Federated', color='tab:blue')
    axs[1].plot(rounds, cent_times, marker='x', label='Centralized', color='tab:orange')
    axs[1].set_title("Time per Round/Epoch", fontsize=12)
    axs[1].set_xlabel("Round/Epoch", fontsize=10)
    axs[1].set_ylabel("Time (s)", fontsize=10)
    axs[1].legend()
    axs[1].grid(True)

    # Add overall title
    fig.suptitle("Comparison: Federated Learning vs Traditional ML", fontsize=14)

    # Embed in Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    root.mainloop()