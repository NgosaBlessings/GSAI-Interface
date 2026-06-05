import customtkinter as ctk
from logic_engine import GSAILogic
import numpy as np

ctk.set_appearance_mode("Dark")

class GSAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("GSAI - Gamified Skill Acquisition Interface")
        self.geometry("1100x630")

        # Track parameters and stats
        self.iteration_count = 0
        self.user_coords = []
        self.k_value = 3  # Default cluster target
        self.is_converged = False

        # Grid config
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar Panel
        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="GSAI", font=("Arial", 24, "bold")).pack(pady=15)

        # NEW: Dynamic K Selection Slider
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

        # Stats Labels
        self.score_label = ctk.CTkLabel(self.sidebar, text="Score: 0.00", font=("Arial", 14))
        self.score_label.pack(pady=5)

        self.iter_label = ctk.CTkLabel(self.sidebar, text="Iteration: 0", font=("Arial", 14))
        self.iter_label.pack(pady=5)

        # NEW: Convergence Visual Status Alert Text
        self.status_label = ctk.CTkLabel(self.sidebar, text="Status: Placing Centroids", text_color="#FF5F1F", font=("Arial", 13, "bold"))
        self.status_label.pack(pady=10)

        # Interactive Canvas Area
        self.canvas = ctk.CTkCanvas(self, bg="#1a1a1a", highlightthickness=0)
        self.canvas.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.canvas.bind("<Button-1>", self.handle_click)

        # Initialize math engine
        self.engine = GSAILogic('data/KDDTrain+_20Percent.txt')

    def change_k_value(self, value):
        """Updates the chosen K value dynamic constraint from user input slider"""
        self.k_value = int(value)
        self.k_label.configure(text=f"Target Clusters (K): {self.k_value}")
        self.reset_interface()

    def draw_points(self):
        self.reset_interface()
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        for p in self.engine.points:
            self.canvas.create_oval(p[0]*w-2, p[1]*h-2, p[0]*w+2, p[1]*h+2, fill="#39FF14", outline="")

    def handle_click(self, event):
        # Limit clicks to exactly the user-defined dynamic target value K
        if len(self.user_coords) < self.k_value:
            x, y = event.x, event.y
            self.canvas.create_text(x, y, text="+", fill="#FF5F1F", font=("Arial", 20, "bold"))
            self.user_coords.append([x/self.canvas.winfo_width(), y/self.canvas.winfo_height()])
            
            # Update user click status instructions
            remaining = self.k_value - len(self.user_coords)
            if remaining > 0:
                self.status_label.configure(text=f"Place {remaining} more '+'", text_color="#FF5F1F")
            else:
                self.status_label.configure(text="Ready to Step!", text_color="#39FF14")
                self.btn_step.configure(state="normal")
                # Lock slider while calculation is hot
                self.k_slider.configure(state="disabled")

    def next_logic_step(self):
        if self.is_converged:
            return

        if not isinstance(self.engine.centroids, np.ndarray):
            self.engine.centroids = np.array(self.user_coords)
        
        self.engine.step_calculate_clusters()
        self.refresh_canvas()
        
        # Move centroids and query absolute convergence state
        converged = self.engine.step_move_centroids()
        
        self.iteration_count += 1
        score = self.engine.calculate_wcss()
        
        self.score_label.configure(text=f"Score: {score:.2f}")
        self.iter_label.configure(text=f"Iteration: {self.iteration_count}")

        # NEW: Convergence visual handler alert trigger
        if converged and self.iteration_count > 1:
            self.is_converged = True
            self.status_label.configure(text="Status: CONVERGED (Done!)", text_color="#39FF14")
            self.iter_label.configure(text_color="#39FF14")
            self.btn_step.configure(state="disabled")
        else:
            self.status_label.configure(text="Running Calculations...", text_color="#1F51FF")

    def refresh_canvas(self):
        self.canvas.delete("all")
        colors = ["#FF3131", "#1F51FF", "#BC13FE", "#FFF01F", "#00FFFF"]
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()

        for i, p in enumerate(self.engine.points):
            c_idx = self.engine.clusters[i]
            color = colors[c_idx % len(colors)]
            self.canvas.create_oval(p[0]*w-2, p[1]*h-2, p[0]*w+2, p[1]*h+2, fill=color, outline="")

        for c in self.engine.centroids:
            self.canvas.create_text(c[0]*w, c[1]*h, text="+", fill="white", font=("Arial", 22, "bold"))

    def reset_interface(self):
        """Clears memory vectors, unlocks control parameters, and resets initial visual elements"""
        self.iteration_count = 0
        self.user_coords = []
        self.is_converged = False
        self.engine.centroids = []
        self.engine.clusters = []
        
        # Reset labels back to default look ahead states
        self.iter_label.configure(text="Iteration: 0", text_color="white")
        self.score_label.configure(text="Score: 0.00")
        self.status_label.configure(text=f"Place {self.k_value} Centroids", text_color="#FF5F1F")
        
        self.btn_step.configure(state="disabled")
        self.k_slider.configure(state="normal")
        
        self.canvas.delete("all")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if len(self.engine.points) > 0:
            for p in self.engine.points:
                self.canvas.create_oval(p[0]*w-2, p[1]*h-2, p[0]*w+2, p[1]*h+2, fill="#39FF14", outline="")

if __name__ == "__main__":
    app = GSAIApp()
    app.mainloop()