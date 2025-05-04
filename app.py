import threading
import tkinter as tk
from tkinter import ttk, IntVar, StringVar
from server import simulate_federated_learning, run_multiple_schemes  # Update import
from gui import show_comparison  # Update import
from comparison_utils import plot_multi_scheme_comparison  # Update import
from dp_utils import PrivacyScheme  # Update import

class FLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FL Simulator")
        self.geometry("300x260")  # Reduced height after removing checkbox
        self.clients = IntVar(value=5)
        self.rounds = IntVar(value=10)
        self.dp_scheme = StringVar(value=PrivacyScheme.GAUSSIAN)

        # --- UI Elements ---
        frame = ttk.Frame(self, padding=10)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="# Clients:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(frame, from_=1, to=100, textvariable=self.clients).grid(row=0, column=1)

        ttk.Label(frame, text="# Rounds:").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(frame, from_=1, to=100, textvariable=self.rounds).grid(row=1, column=1)

        # Privacy scheme selection
        ttk.Label(frame, text="Privacy Scheme:").grid(row=2, column=0, sticky="w")
        scheme_combo = ttk.Combobox(frame, textvariable=self.dp_scheme, state="readonly")
        scheme_combo["values"] = [
            PrivacyScheme.GAUSSIAN,
            PrivacyScheme.LAPLACE
        ]
        scheme_combo.grid(row=2, column=1)

        self.start_btn = ttk.Button(frame, text="Start Simulation", command=self.start_sim)
        self.start_btn.grid(row=3, columnspan=2, pady=10)
        
        self.status = ttk.Label(frame, text="Ready", foreground="green")
        self.status.grid(row=4, columnspan=2)

        self.compare_btn = ttk.Button(
            frame, 
            text="Compare All Schemes", 
            command=self.run_comparison
        )
        self.compare_btn.grid(row=5, columnspan=2, pady=10)

    def start_sim(self):
        self.start_btn.config(state="disabled")
        self.status.config(text="Running...", foreground="blue")
        threading.Thread(target=self.run_and_plot, daemon=True).start()

    def run_and_plot(self):
        try:
            n = self.clients.get()
            r = self.rounds.get()
            scheme = self.dp_scheme.get()
            
            # Run simulation
            fl_accs, fl_times, cent_accs, cent_times = simulate_federated_learning(
                n, r, dp_enabled=True, dp_scheme=scheme
            )

            self.status.config(text="Done", foreground="green")
            # Schedule GUI update in main thread
            self.after(100, lambda: show_comparison(fl_accs, fl_times, cent_accs, cent_times))
            self.start_btn.config(state="normal")
        except Exception as e:
            self.status.config(text=f"Error: {str(e)}", foreground="red")
            self.start_btn.config(state="normal")

    def run_comparison(self):
        """Run comparison of all privacy schemes"""
        self.compare_btn.config(state="disabled")
        self.start_btn.config(state="disabled")
        self.status.config(text="Running comparisons...", foreground="blue")
        threading.Thread(target=self._run_and_plot_comparison, daemon=True).start()

    def _run_and_plot_comparison(self):
        try:
            n = self.clients.get()
            r = self.rounds.get()
            
            results = run_multiple_schemes(n, r)
            plot_multi_scheme_comparison(results)
            
            self.status.config(text="Comparison complete", foreground="green")
        except Exception as e:
            self.status.config(text=f"Error: {str(e)}", foreground="red")
        finally:
            self.compare_btn.config(state="normal")
            self.start_btn.config(state="normal")

if __name__ == "__main__":
    app = FLApp()
    app.mainloop()
