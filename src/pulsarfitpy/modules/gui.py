import threading
import tkinter as tk
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

from tkinter import ttk, scrolledtext, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from psrqpy import QueryATNF
from .approximation import PulsarApproximation
from .utils import configure_logging


class PulsarFitGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("pulsarfitpy - Pulsar Data Analysis")
        self.root.geometry("1200x800")

        configure_logging("ERROR")

        self.approx = None
        self.query = None

        self._create_widgets()

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)

        self._create_parameter_section(left_frame)
        self._create_results_section(right_frame)

    def _create_parameter_section(self, parent):
        param_frame = ttk.LabelFrame(parent, text="Parameters", padding="10")
        param_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), pady=5)

        ttk.Label(param_frame, text="X Parameter:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.x_param_var = tk.StringVar(value="P0")
        x_param_entry = ttk.Entry(param_frame, textvariable=self.x_param_var, width=20)
        x_param_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)

        ttk.Label(param_frame, text="Y Parameter:").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.y_param_var = tk.StringVar(value="P1")
        y_param_entry = ttk.Entry(param_frame, textvariable=self.y_param_var, width=20)
        y_param_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)

        ttk.Label(param_frame, text="Max Degree:").grid(
            row=2, column=0, sticky=tk.W, pady=2
        )
        self.degree_var = tk.IntVar(value=5)
        degree_spin = ttk.Spinbox(
            param_frame, from_=1, to=10, textvariable=self.degree_var, width=18
        )
        degree_spin.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2)

        self.log_x_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Log X", variable=self.log_x_var).grid(
            row=3, column=0, sticky=tk.W, pady=2
        )

        self.log_y_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Log Y", variable=self.log_y_var).grid(
            row=3, column=1, sticky=tk.W, pady=2
        )

        ttk.Label(param_frame, text="ATNF Condition:").grid(
            row=4, column=0, sticky=tk.W, pady=2
        )
        self.condition_var = tk.StringVar(value="")
        condition_entry = ttk.Entry(
            param_frame, textvariable=self.condition_var, width=20
        )
        condition_entry.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=2)

        btn_frame = ttk.Frame(param_frame)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=10)

        self.fit_btn = ttk.Button(
            btn_frame, text="Fit Polynomial", command=self._on_fit
        )
        self.fit_btn.grid(row=0, column=0, padx=5)

        self.query_btn = ttk.Button(
            btn_frame, text="Query ATNF", command=self._on_query
        )
        self.query_btn.grid(row=0, column=1, padx=5)

        self.status_label = ttk.Label(param_frame, text="Ready", foreground="blue")
        self.status_label.grid(row=6, column=0, columnspan=2, pady=5)

        log_frame = ttk.LabelFrame(parent, text="Output Log", padding="10")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        parent.rowconfigure(1, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, width=40)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

    def _create_results_section(self, parent):
        plot_frame = ttk.LabelFrame(parent, text="Visualization", padding="10")
        plot_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(
            row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)
        )
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

    def _log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def _on_query(self):
        def query_thread():
            try:
                self.status_label.config(text="Querying ATNF...", foreground="orange")
                self.query_btn.config(state="disabled")

                x_param = self.x_param_var.get()
                y_param = self.y_param_var.get()
                condition = self.condition_var.get() or None

                self._log(f"\nQuerying ATNF for {x_param} and {y_param}...")

                params = [x_param, y_param]
                self.query = QueryATNF(params=params, condition=condition)

                num_pulsars = len(self.query.table)
                self._log(f"Found {num_pulsars} pulsars")

                for param in params:
                    if param in self.query.table.colnames:
                        data = self.query.table[param].data
                        valid = np.isfinite(data).sum()
                        self._log(f"  {param}: {valid} valid values")

                self.status_label.config(text="Query complete", foreground="green")

            except Exception as e:
                self._log(f"Error: {str(e)}")
                self.status_label.config(text="Query failed", foreground="red")
                messagebox.showerror("Query Error", str(e))
            finally:
                self.query_btn.config(state="normal")

        thread = threading.Thread(target=query_thread, daemon=True)
        thread.start()

    def _on_fit(self):
        def fit_thread():
            try:
                self.status_label.config(
                    text="Fitting polynomial...", foreground="orange"
                )
                self.fit_btn.config(state="disabled")

                x_param = self.x_param_var.get()
                y_param = self.y_param_var.get()
                test_degree = self.degree_var.get()
                log_x = self.log_x_var.get()
                log_y = self.log_y_var.get()
                condition = self.condition_var.get() or None

                self._log(f"\nFitting polynomial (degree={test_degree})...")

                query = QueryATNF(params=[x_param, y_param], condition=condition)

                self.approx = PulsarApproximation(
                    query=query,
                    x_param=x_param,
                    y_param=y_param,
                    test_degree=test_degree,
                    log_x=log_x,
                    log_y=log_y,
                )

                self.approx.fit_polynomial(verbose=False)

                self._log(f"Best degree: {self.approx.best_degree}")
                self._log("R² scores:")
                for deg, score in self.approx.r2_scores.items():
                    self._log(f"  Degree {deg}: {score:.6f}")

                metrics = self.approx.compute_metrics(verbose=False)
                self._log("\nMetrics:")
                self._log(f"  R² = {metrics['r2']:.6f}")
                self._log(f"  RMSE = {metrics['rmse']:.6e}")
                self._log(f"  MAE = {metrics['mae']:.6e}")
                self._log(f"  χ² = {metrics['chi2_reduced']:.6f}")

                self._plot_results()

                self.status_label.config(text="Fit complete", foreground="green")

            except Exception as e:
                self._log(f"Error: {str(e)}")
                self.status_label.config(text="Fit failed", foreground="red")
                messagebox.showerror("Fit Error", str(e))
            finally:
                self.fit_btn.config(state="normal")

        thread = threading.Thread(target=fit_thread, daemon=True)
        thread.start()

    def _plot_results(self):
        self.figure.clear()

        ax1 = self.figure.add_subplot(211)
        ax1.scatter(
            self.approx.x_data, self.approx.y_data, s=10, alpha=0.4, label="Data"
        )
        ax1.plot(
            self.approx.predicted_x,
            self.approx.predicted_y,
            "r-",
            linewidth=2,
            label=f"Degree {self.approx.best_degree}",
        )
        xlabel = (
            f"log({self.approx.x_param})" if self.approx.log_x else self.approx.x_param
        )
        ylabel = (
            f"log({self.approx.y_param})" if self.approx.log_y else self.approx.y_param
        )
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title("Polynomial Fit")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = self.figure.add_subplot(212)
        degrees = list(self.approx.r2_scores.keys())
        scores = list(self.approx.r2_scores.values())
        ax2.plot(degrees, scores, "o-", linewidth=2, markersize=8)
        ax2.set_xlabel("Polynomial Degree")
        ax2.set_ylabel("R² Score")
        ax2.set_title("Model Selection")
        ax2.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()


def launch_gui():
    root = tk.Tk()
    app = PulsarFitGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
