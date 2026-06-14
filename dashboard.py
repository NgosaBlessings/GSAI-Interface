import customtkinter as ctk
from logic_engine import GSAILogic
import numpy as np
import os
from datetime import datetime

ctk.set_appearance_mode("Dark")

class GSAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("GSAI - Gamified Skill Acquisition Interface")
        self.geometry("1100x710") # Expanded to fit export actions neatly

        # Track parameters and stats
        self.iteration_count = 0
        self.user_coords = []
        self.k_value = 3  
        self.is_converged = False
        self.show_optimal = False  

        # Grid config
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar Panel
        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="GSAI", font=("Arial", 24, "bold")).pack(pady=15)

        # Dynamic K Selection Slider
        self.k_label = ctk.CTkLabel(self.sidebar, text=f"Target Clusters (K): {self.k_value}", font=("Arial", 13, "bold"))
        self.k_label.pack(pady=(10, 0))
        
        self.k_slider = ctk.CTkSlider(self.sidebar, from_=2, to=5, number_of_steps=3, command=self.change_k_value)
        self.k_slider.set(self.k_value)
        self.k_slider.pack(pady=(5, 15), padx=20)

        self.btn_init = ctk.CTkButton(self.sidebar, text="1. Initialize Points", command=self.draw_points)
        self.btn_init.pack(pady=10, padx=20)

        self.btn_step = ctk.CTkButton(self.sidebar, text="2. Next Step", state="disabled", command=self.next_logic_step)
        self.btn_step.pack(pady=10, padx=20)

        self.btn_reset = ctk.CTkButton(self.sidebar, text="3. Reset Engine", fg_color="#721c24", hover_color="#a93226", command=self.reset_interface)
        self.btn_reset.pack(pady=10, padx=20)

        self.btn_optimal = ctk.CTkButton(self.sidebar, text="4. Show Optimal Solution", fg_color="#0e6251", hover_color="#117a65", state="disabled", command=self.toggle_optimal_solution)
        self.btn_optimal.pack(pady=10, padx=20)

        # NEW: Export Metrics Data Button
        self.btn_export = ctk.CTkButton(self.sidebar, text="5. Export Session Metrics", fg_color="#512e5f", hover_color="#6c3483", state="disabled", command=self.export_session_data)
        self.btn_export.pack(pady=10, padx=20)

        # Stats Labels
        self.score_label = ctk.CTkLabel(self.sidebar, text="Score: 0.00", font=("Arial", 14))
        self.score_label.pack(pady=5)

        self.iter_label = ctk.CTkLabel(self.sidebar, text="Iteration: 0", font=("Arial", 14))
        self.iter_label.pack(pady=5)

        # Convergence Visual Status Alert Text
        self.status_label = ctk.CTkLabel(self.sidebar, text="Status: Placing Centroids", text_color="#FF5F1F", font=("Arial", 13, "bold"))
        self.status_label.pack(pady=10)

        # Interactive Canvas Area
        self.canvas = ctk.CTkCanvas(self, bg="#1a1a1a", highlightthickness=0)
        self.canvas.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.canvas.bind("<Button-1>", self.handle_click)

        # Initialize math engine
        self.engine = GSAILogic('data/KDDTrain+_20Percent.txt')

    def change_k_value(self, value):
        self.k_value = int(value)
        self.k_label.configure(text=f"Target Clusters (K): {self.k_value}")
        self.reset_interface()

    def draw_points(self):
        self.reset_interface()
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        for p in self.engine.points:
            self.canvas.create_oval(p[0]*w-2, p[1]*h-2, p[0]*w+2, p[1]*h+2, fill="#39FF14", outline="")

    def handle_click(self, event):
        if len(self.user_coords) < self.k_value:
            x, y = event.x, event.y
            self.canvas.create_text(x, y, text="+", fill="#FF5F1F", font=("Arial", 20, "bold"))
            self.user_coords.append([x/self.canvas.winfo_width(), y/self.canvas.winfo_height()])
            
            remaining = self.k_value - len(self.user_coords)
            if remaining > 0:
                self.status_label.configure(text=f"Place {remaining} more '+'", text_color="#FF5F1F")
            else:
                self.status_label.configure(text="Ready to Step!", text_color="#39FF14")
                self.btn_step.configure(state="normal")
                self.btn_optimal.configure(state="normal")  
                self.k_slider.configure(state="disabled")

    def toggle_optimal_solution(self):
        self.show_optimal = not self.show_optimal
        if self.show_optimal:
            self.btn_optimal.configure(text="4. Hide Optimal Solution", fg_color="#b03a2e", hover_color="#922b21")
        else:
            self.btn_optimal.configure(text="4. Show Optimal Solution", fg_color="#0e6251", hover_color="#117a65")
        if len(self.engine.clusters) > 0:
            self.refresh_canvas()

    def next_logic_step(self):
        if self.is_converged:
            return

        if not isinstance(self.engine.centroids, np.ndarray):
            self.engine.centroids = np.array(self.user_coords)
        
        self.engine.step_calculate_clusters()
        self.refresh_canvas()
        
        converged = self.engine.step_move_centroids()
        
        self.iteration_count += 1
        score = self.engine.calculate_wcss()
        
        self.score_label.configure(text=f"Score: {score:.2f}")
        self.iter_label.configure(text=f"Iteration: {self.iteration_count}")

        if converged and self.iteration_count > 1:
            self.is_converged = True
            self.status_label.configure(text="Status: CONVERGED (Done!)", text_color="#39FF14")
            self.iter_label.configure(text_color="#39FF14")
            self.btn_step.configure(state="disabled")
            self.btn_export.configure(state="normal")  # NEW: Unlock file export on safe convergence
        else:
            self.status_label.configure(text="Running Calculations...", text_color="#1F51FF")

    def export_session_data(self):
        """NEW: Generates an analytical metrics report file for thesis logging"""
        if not self.is_converged:
            return
            
        os.makedirs("exports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exports/gsai_session_{timestamp}.csv"
        
        user_score = self.engine.calculate_wcss()
        opt_score = self.engine.get_optimal_wcss(self.k_value)
        
        # Skill acquisition calculation: how close is the user's manual optimization to the absolute machine minimum?
        accuracy = min(100.0, (opt_score / user_score) * 100) if user_score > 0 else 0
        
        labels = self.engine.get_cluster_labels()
        
        with open(filename, "w") as f:
            f.write("=== GSAI SYSTEM METRICS RESEARCH EXPORT ===\n")
            f.write(f"Timestamp Logged,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Target Clusters Selected (K),{self.k_value}\n")
            f.write(f"Total Algorithmic Iterations,{self.iteration_count}\n")
            f.write(f"User Final WCSS Score,{user_score:.4f}\n")
            f.write(f"Theoretical Optimal WCSS Score,{opt_score:.4f}\n")
            f.write(f"User Optimization Accuracy Profile,{accuracy:.2f}%\n\n")
            
            f.write("--- Identified Cluster Profiles ---\n")
            f.write("Cluster Index,Centroid X Coordinate,Centroid Y Coordinate,Network Class Identity\n")
            for idx, c in enumerate(self.engine.centroids):
                f.write(f"{idx},{c[0]:.4f},{c[1]:.4f},{labels[idx]}\n")
                
        self.status_label.configure(text="Metrics Saved to /exports!", text_color="#00FFFF")
        self.btn_export.configure(state="disabled")

    def refresh_canvas(self):
        self.canvas.delete("all")
        colors = ["#FF3131", "#1F51FF", "#BC13FE", "#FFF01F", "#00FFFF"]
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()

        labels = self.engine.get_cluster_labels()

        for i, p in enumerate(self.engine.points):
            c_idx = self.engine.clusters[i]
            color = colors[c_idx % len(colors)]
            self.canvas.create_oval(p[0]*w-2, p[1]*h-2, p[0]*w+2, p[1]*h+2, fill=color, outline="")

        for idx, c in enumerate(self.engine.centroids):
            cx, cy = c[0]*w, c[1]*h
            self.canvas.create_text(cx, cy, text="+", fill="white", font=("Arial", 22, "bold"))
            
            lbl_text = labels[idx]
            lbl_color = "#39FF14" if "Normal" in lbl_text else "#FF3131"
            self.canvas.create_text(cx + 15, cy - 15, text=lbl_text, fill=lbl_color, font=("Arial", 11, "bold"), anchor="w")

        if self.show_optimal:
            opt_centroids = self.engine.get_optimal_centroids(self.k_value)
            for oc in opt_centroids:
                ocx, ocy = oc[0]*w, oc[1]*h
                self.canvas.create_oval(ocx-12, ocy-12, ocx+12, ocy+12, outline="white", width=3)
                self.canvas.create_text(ocx, ocy, text="★", fill="white", font=("Arial", 10, "bold"))

    def reset_interface(self):
        self.iteration_count = 0
        self.user_coords = []
        self.is_converged = False
        self.show_optimal = False
        self.engine.centroids = []
        self.engine.clusters = []
        
        self.iter_label.configure(text="Iteration: 0", text_color="white")
        self.score_label.configure(text="Score: 0.00")
        self.status_label.configure(text=f"Place {self.k_value} Centroids", text_color="#FF5F1F")
        
        self.btn_step.configure(state="disabled")
        self.btn_optimal.configure(text="4. Show Optimal Solution", fg_color="#0e6251", hover_color="#117a65", state="disabled")
        self.btn_export.configure(state="disabled") # Re-lock export on reset
        self.k_slider.configure(state="normal")
        
        self.canvas.delete("all")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if len(self.engine.points) > 0:
            for p in self.engine.points:
                self.canvas.create_oval(p[0]*w-2, p[1]*h-2, p[0]*w+2, p[1]*h+2, fill="#39FF14", outline="")

if __name__ == "__main__":
    app = GSAIApp()
    app.mainloop()