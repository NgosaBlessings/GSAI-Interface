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
        self.geometry("1150x730")

        # Track parameters and stats
        self.iteration_count = 0
        self.user_coords = []
        self.current_level = 1
        self.k_value = 2  # Default K for Level 1
        self.is_converged = False
        self.show_optimal = False  

        # Level configuration presets
        self.level_k_map = {1: 2, 2: 3, 3: 3, 4: 4, 5: 5}

        # Grid config
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar Panel
        self.sidebar = ctk.CTkFrame(self, width=220)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="GSAI", font=("Arial", 24, "bold")).pack(pady=10)

        # Level Selector Dropdown
        ctk.CTkLabel(self.sidebar, text="Select Game Level:", font=("Arial", 12, "bold")).pack(pady=(5, 0))
        self.level_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=[
                "Level 1: Basics of Denial (K=2)",
                "Level 2: Web Server Attacks (K=3)",
                "Level 3: Network Probing (K=3)",
                "Level 4: Multi-Vector Threat (K=4)",
                "Level 5: Complex Intrusion (K=5)"
            ],
            command=self.on_level_change
        )
        self.level_menu.pack(pady=(5, 10), padx=15)

        # Dynamic K Selection Slider
        self.k_label = ctk.CTkLabel(self.sidebar, text=f"Target Clusters (K): {self.k_value}", font=("Arial", 13, "bold"))
        self.k_label.pack(pady=(5, 0))
        
        self.k_slider = ctk.CTkSlider(self.sidebar, from_=2, to=5, number_of_steps=3, command=self.change_k_value)
        self.k_slider.set(self.k_value)
        self.k_slider.pack(pady=(5, 10), padx=15)

        # Tutorial / User Guide Button
        self.btn_guide = ctk.CTkButton(self.sidebar, text="📖 How to Use / Guide", fg_color="#1f618d", hover_color="#154360", command=self.show_tutorial)
        self.btn_guide.pack(pady=(0, 10), padx=15)

        self.btn_init = ctk.CTkButton(self.sidebar, text="1. Initialize Points", command=self.draw_points)
        self.btn_init.pack(pady=6, padx=15)

        self.btn_step = ctk.CTkButton(self.sidebar, text="2. Next Step", state="disabled", command=self.next_logic_step)
        self.btn_step.pack(pady=6, padx=15)

        self.btn_reset = ctk.CTkButton(self.sidebar, text="3. Reset Engine", fg_color="#721c24", hover_color="#a93226", command=self.reset_interface)
        self.btn_reset.pack(pady=6, padx=15)

        self.btn_optimal = ctk.CTkButton(self.sidebar, text="4. Show Optimal Solution", fg_color="#0e6251", hover_color="#117a65", state="disabled", command=self.toggle_optimal_solution)
        self.btn_optimal.pack(pady=6, padx=15)

        # Export Metrics Data Button
        self.btn_export = ctk.CTkButton(self.sidebar, text="5. Export Session Metrics", fg_color="#512e5f", hover_color="#6c3483", state="disabled", command=self.export_session_data)
        self.btn_export.pack(pady=6, padx=15)

        # Stats Labels
        self.score_label = ctk.CTkLabel(self.sidebar, text="Score: 0.00", font=("Arial", 14))
        self.score_label.pack(pady=4)

        self.iter_label = ctk.CTkLabel(self.sidebar, text="Iteration: 0", font=("Arial", 14))
        self.iter_label.pack(pady=4)

        # Convergence Visual Status Alert Text
        self.status_label = ctk.CTkLabel(self.sidebar, text="Status: Select Level", text_color="#FF5F1F", font=("Arial", 12, "bold"))
        self.status_label.pack(pady=8)

        # Interactive Canvas Area
        self.canvas = ctk.CTkCanvas(self, bg="#1a1a1a", highlightthickness=0)
        self.canvas.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.canvas.bind("<Button-1>", self.handle_click)

        # Initialize math engine with NSL-KDD dataset
        self.engine = GSAILogic('data/KDDTrain+_20Percent.txt')

    def on_level_change(self, selected_option):
        """Handles switching between game levels and updating dataset plots."""
        level_num = int(selected_option.split(":")[0].replace("Level", "").strip())
        self.current_level = level_num
        self.k_value = self.level_k_map[level_num]
        
        # Update slider UI
        self.k_slider.set(self.k_value)
        self.k_label.configure(text=f"Target Clusters (K): {self.k_value}")
        
        # Load new level sample data in engine
        self.engine.load_level(level_num)
        self.reset_interface()
        self.draw_points()

    def change_k_value(self, value):
        self.k_value = int(value)
        self.k_label.configure(text=f"Target Clusters (K): {self.k_value}")
        self.reset_interface()

    def show_convergence_popup(self):
        """Displays a clean auto-closing toast popup when convergence is reached."""
        popup = ctk.CTkToplevel(self)
        popup.title("")
        popup.geometry("380x130")
        popup.resizable(False, False)
        popup.attributes("-topmost", True)
        
        # Center popup over main app window
        app_x = self.winfo_x() + (self.winfo_width() // 2) - 190
        app_y = self.winfo_y() + (self.winfo_height() // 2) - 65
        popup.geometry(f"+{app_x}+{app_y}")

        frame = ctk.CTkFrame(popup, fg_color="#1c2833", border_color="#27ae60", border_width=2)
        frame.pack(fill="both", expand=True, padx=5, pady=5)

        ctk.CTkLabel(frame, text=f"🎉 Level {self.current_level} Converged!", font=("Arial", 16, "bold"), text_color="#27ae60").pack(pady=(20, 5))
        ctk.CTkLabel(frame, text="Centroids have stabilized. Algorithm complete.", font=("Arial", 12), text_color="#ecf0f1").pack()
        ctk.CTkLabel(frame, text="(This window closes automatically in 3 seconds)", font=("Arial", 10, "italic"), text_color="#85929e").pack(pady=(5, 0))

        # Auto close after 3 seconds
        self.after(3000, popup.destroy)

    def show_tutorial(self):
        """Opens a modal guiding new users through the level progression."""
        guide = ctk.CTkToplevel(self)
        guide.title("GSAI Interface Guide")
        guide.geometry("520x440")
        guide.resizable(False, False)
        guide.attributes("-topmost", True)

        app_x = self.winfo_x() + (self.winfo_width() // 2) - 260
        app_y = self.winfo_y() + (self.winfo_height() // 2) - 220
        guide.geometry(f"+{app_x}+{app_y}")

        frame = ctk.CTkFrame(guide, fg_color="#1a1a1a")
        frame.pack(fill="both", expand=True, padx=15, pady=15)

        ctk.CTkLabel(frame, text="How to Play GSAI", font=("Arial", 18, "bold"), text_color="#00FFFF").pack(pady=(10, 15))

        guide_steps = [
            "1. Choose a Level from the dropdown menu (Level 1 to 5).",
            "2. Click '1. Initialize Points' to load the traffic dataset layout.",
            "3. Click directly on the black canvas area to place your K '+' centroids.",
            "4. Click '2. Next Step' repeatedly to watch clusters form and move.",
            "5. When centroids stabilize, compare with '4. Show Optimal Solution'.",
            "6. Click '5. Export Session Metrics' to save your results to CSV."
        ]

        for step in guide_steps:
            ctk.CTkLabel(frame, text=step, font=("Arial", 12), anchor="w", justify="left").pack(fill="x", padx=15, pady=5)

        ctk.CTkButton(frame, text="Got It!", fg_color="#27ae60", hover_color="#1e8449", command=guide.destroy).pack(pady=20)

    def draw_points(self):
        """Loads initial dataset points as neutral white (#FFFFFF)."""
        self.reset_interface()
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        for p in self.engine.points:
            self.canvas.create_oval(p[0]*w-2, p[1]*h-2, p[0]*w+2, p[1]*h+2, fill="#FFFFFF", outline="")

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
                self.level_menu.configure(state="disabled")

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
            self.btn_export.configure(state="normal")
            
            # Trigger auto-closing popup
            self.show_convergence_popup()
        else:
            self.status_label.configure(text="Running Calculations...", text_color="#1F51FF")

    def export_session_data(self):
        if not self.is_converged:
            return
            
        os.makedirs("exports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exports/gsai_level{self.current_level}_{timestamp}.csv"
        
        user_score = self.engine.calculate_wcss()
        opt_score = self.engine.get_optimal_wcss(self.k_value)
        
        accuracy = min(100.0, (opt_score / user_score) * 100) if user_score > 0 else 0
        labels = self.engine.get_cluster_labels()
        
        with open(filename, "w") as f:
            f.write("=== GSAI SYSTEM METRICS RESEARCH EXPORT ===\n")
            f.write(f"Timestamp Logged,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Active Game Level,{self.current_level}\n")
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
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()

        labels, colors = self.engine.get_cluster_labels_and_colors()

        # Draw Points
        for i, p in enumerate(self.engine.points):
            c_idx = self.engine.clusters[i]
            point_color = colors[c_idx] if c_idx < len(colors) else "#FFFFFF"
            self.canvas.create_oval(p[0]*w-2, p[1]*h-2, p[0]*w+2, p[1]*h+2, fill=point_color, outline="")

        # Draw Centroids & Labels
        for idx, c in enumerate(self.engine.centroids):
            cx, cy = c[0]*w, c[1]*h
            c_color = colors[idx] if idx < len(colors) else "#FFFFFF"
            lbl_text = labels[idx] if idx < len(labels) else ""
            
            self.canvas.create_text(cx, cy, text="+", fill="white", font=("Arial", 22, "bold"))
            self.canvas.create_text(cx + 15, cy - 15, text=lbl_text, fill=c_color, font=("Arial", 11, "bold"), anchor="w")

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
        self.btn_export.configure(state="disabled")
        self.k_slider.configure(state="normal")
        self.level_menu.configure(state="normal")
        
        self.canvas.delete("all")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if len(self.engine.points) > 0:
            for p in self.engine.points:
                self.canvas.create_oval(p[0]*w-2, p[1]*h-2, p[0]*w+2, p[1]*h+2, fill="#FFFFFF", outline="")

if __name__ == "__main__":
    app = GSAIApp()
    app.mainloop()