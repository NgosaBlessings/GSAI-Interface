import customtkinter as ctk
from logic_engine import GSAILogic
import numpy as np

ctk.set_appearance_mode("Dark")

class GSAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("GSAI - Gamified Skill Acquisition Interface")
        self.geometry("1100x600")

        # Grid config
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="GSAI", font=("Arial", 24, "bold")).pack(pady=20)

        self.btn_init = ctk.CTkButton(self.sidebar, text="1. Initialize Points", command=self.draw_points)
        self.btn_init.pack(pady=10, padx=20)

        self.btn_step = ctk.CTkButton(self.sidebar, text="2. Next Step", state="disabled", command=self.next_logic_step)
        self.btn_step.pack(pady=10, padx=20)

        self.score_label = ctk.CTkLabel(self.sidebar, text="Score: 0.00", font=("Arial", 14))
        self.score_label.pack(pady=20)

        # Canvas
        self.canvas = ctk.CTkCanvas(self, bg="#1a1a1a", highlightthickness=0)
        self.canvas.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        # Bind CLICK to canvas
        self.canvas.bind("<Button-1>", self.handle_click)

        # Logic Setup
        self.engine = GSAILogic('data/KDDTrain+_20Percent.txt')
        self.user_coords = []

    def draw_points(self):
        self.canvas.delete("all")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        for p in self.engine.points:
            self.canvas.create_oval(p[0]*w-2, p[1]*h-2, p[0]*w+2, p[1]*h+2, fill="#39FF14", outline="")
        print("Points rendered.")

    def handle_click(self, event):
        if len(self.user_coords) < 5:
            x, y = event.x, event.y
            self.canvas.create_text(x, y, text="+", fill="#FF5F1F", font=("Arial", 20, "bold"))
            
            # Normalize and save
            self.user_coords.append([x/self.canvas.winfo_width(), y/self.canvas.winfo_height()])
            self.btn_step.configure(state="normal")

    def next_logic_step(self):
        if not self.engine.centroids:
            self.engine.centroids = np.array(self.user_coords)
        
        self.engine.step_calculate_clusters()
        self.refresh_canvas()
        self.engine.step_move_centroids()
        
        score = self.engine.calculate_wcss()
        self.score_label.configure(text=f"Score: {score:.2f}")

    def refresh_canvas(self):
        self.canvas.delete("all")
        colors = ["#FF3131", "#39FF14", "#1F51FF", "#FFF01F", "#BC13FE"]
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()

        for i, p in enumerate(self.engine.points):
            c_idx = self.engine.clusters[i]
            color = colors[c_idx % len(colors)]
            self.canvas.create_oval(p[0]*w-2, p[1]*h-2, p[0]*w+2, p[1]*h+2, fill=color, outline="")

        for c in self.engine.centroids:
            self.canvas.create_text(c[0]*w, c[1]*h, text="+", fill="white", font=("Arial", 22, "bold"))

if __name__ == "__main__":
    app = GSAIApp()
    app.mainloop()