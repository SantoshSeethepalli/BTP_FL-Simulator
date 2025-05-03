import threading
import tkinter as tk
from tkinter import ttk, BooleanVar, IntVar
from server import simulate_federated_learning
from gui import show_comparison  # New comparison plotting function

class FLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FL Simulator")
        self.geometry("300x220")
        self.clients = IntVar(value=5)
        self.rounds  = IntVar(value=10)
        self.dp      = BooleanVar(value=False)

        # --- UI Elements ---
        frame = ttk.Frame(self, padding=10)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="# Clients:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(frame, from_=1, to=100, textvariable=self.clients).grid(row=0, column=1)

        ttk.Label(frame, text="# Rounds:").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(frame, from_=1, to=100, textvariable=self.rounds).grid(row=1, column=1)

        ttk.Checkbutton(frame, text="Enable DP", variable=self.dp).grid(row=2, columnspan=2, pady=5)

        self.start_btn = ttk.Button(frame, text="Start Simulation", command=self.start_sim)
        self.start_btn.grid(row=3, columnspan=2, pady=10)
        
        self.status = ttk.Label(frame, text="Ready", foreground="green")
        self.status.grid(row=4, columnspan=2)

    def start_sim(self):
        self.start_btn.config(state="disabled")
        self.status.config(text="Running...", foreground="blue")
        threading.Thread(target=self.run_and_plot, daemon=True).start()

    def run_and_plot(self):
        n = self.clients.get()
        r = self.rounds.get()
        use_dp = self.dp.get()

        # Run simulation
        fl_accs, fl_times, cent_accs, cent_times = simulate_federated_learning(
            n, r, dp_enabled=use_dp
        )

        self.status.config(text="Done", foreground="green")
        show_comparison(fl_accs, fl_times, cent_accs, cent_times)
        self.start_btn.config(state="normal")

if __name__ == "__main__":
    app = FLApp()
    app.mainloop()
