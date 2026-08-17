import os
import customtkinter as ctk

# Set global appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class GSAILauncher(ctk.CTk):

  def __init__(self):
    super().__init__()

    # Window Configuration
    self.title("GSAI - AI Skill Acquisition Framework")
    self.geometry("700x550")
    self.resizable(False, False)

    # Main Frame
    self.main_frame = ctk.CTkFrame(self, corner_radius=15)
    self.main_frame.pack(padx=30, pady=30, fill="both", expand=True)

    # Header Title & Subtitle
    self.title_label = ctk.CTkLabel(
        self.main_frame,
        text="GSAI Framework",
        font=ctk.CTkFont(size=28, weight="bold"),
        text_color="#00E5FF",
    )
    self.title_label.pack(pady=(25, 5))

    self.subtitle_label = ctk.CTkLabel(
        self.main_frame,
        text="Interactive Artificial Intelligence Skill Acquisition Platform",
        font=ctk.CTkFont(size=14, slant="italic"),
        text_color="#A0A0A0",
    )
    self.subtitle_label.pack(pady=(0, 20))

    # Divider Line
    self.divider = ctk.CTkFrame(
        self.main_frame, height=2, fg_color="#333333", width=580
    )
    self.divider.pack(pady=10)

    # Instruction Text
    self.select_label = ctk.CTkLabel(
        self.main_frame,
        text="Select a Learning Module to Launch:",
        font=ctk.CTkFont(size=16, weight="bold"),
    )
    self.select_label.pack(pady=(10, 15))

    # Module 1: K-Means Card
    self.btn_kmeans = ctk.CTkButton(
        self.main_frame,
        text=(
            "Module 1: Unsupervised K-Means Clustering\n"
            "• Learn Centroid Placement, WCSS & Convergence\n"
            "• NSL-KDD Traffic Pattern Discovery"
        ),
        font=ctk.CTkFont(size=13),
        height=75,
        corner_radius=10,
        fg_color="#1F2937",
        hover_color="#374151",
        border_width=1,
        border_color="#00E5FF",
        anchor="w",
        command=self.launch_kmeans,
    )
    self.btn_kmeans.pack(padx=40, pady=10, fill="x")

    # Module 2: Decision Tree Card
    self.btn_dtree = ctk.CTkButton(
        self.main_frame,
        text=(
            "Module 2: Supervised Decision Tree Splitting\n"
            "• Learn Axis-Parallel Partitioning & Gini Impurity\n"
            "• NSL-KDD Attack Classification Rules"
        ),
        font=ctk.CTkFont(size=13),
        height=75,
        corner_radius=10,
        fg_color="#1F2937",
        hover_color="#374151",
        border_width=1,
        border_color="#FF9800",
        anchor="w",
        command=self.launch_decision_tree,
    )
    self.btn_dtree.pack(padx=40, pady=10, fill="x")

    # Footer Status
    self.footer_label = ctk.CTkLabel(
        self.main_frame,
        text="System Telemetry: Active | Research Mode: On",
        font=ctk.CTkFont(size=11),
        text_color="#6B7280",
    )
    self.footer_label.pack(side="bottom", pady=15)

  def launch_kmeans(self):
    """Launches the existing K-Means dashboard."""
    try:
      import dashboard

      # If dashboard has a runner function or app instance
      if hasattr(dashboard, "GSAIApp"):
        app_win = dashboard.GSAIApp()
        app_win.focus()
      else:
        os.system("python dashboard.py")
    except Exception as e:
      print(f"[GSAI] Launching dashboard.py via OS system fallback... ({e})")
      os.system("python dashboard.py")

  def launch_decision_tree(self):
    """Launches the new Decision Tree Module."""
    try:
      import decision_tree_module

      decision_tree_module.run_decision_tree_app(parent=self)
    except Exception as e:
      print(f"[GSAI] Error launching decision_tree_module: {e}")


if __name__ == "__main__":
  app = GSAILauncher()
  app.mainloop()