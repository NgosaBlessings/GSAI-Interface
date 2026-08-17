import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Set dark appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class DecisionTreeModule(ctk.CTkToplevel):

  def __init__(self, parent=None):
    super().__init__(parent)

    self.title("GSAI - Module 2: Decision Tree Boundary Splitting")
    self.geometry("1200x750")
    self.minsize(1100, 700)

    # Current Level State (1, 2, or 3)
    self.current_level = 1
    self.v_split = 0.05
    self.h_split = 0.05

    # Generate Initial Dataset for Level 1
    self._load_level_dataset(1)

    # Layout Setup - Enforce strict grid column weighting
    self.grid_columnconfigure(0, weight=0, minsize=320)
    self.grid_columnconfigure(1, weight=1)
    self.grid_rowconfigure(0, weight=1)

    self._build_sidebar()
    self._build_canvas()

    # Show Onboarding Modal on Startup
    self.after(200, self._show_onboarding_modal)

  def _load_level_dataset(self, level):
    """Generates synthetic NSL-KDD data distributions and targets based on the active level."""
    self.current_level = level
    np.random.seed(42 + level)

    if level == 1:
      # Level 1: Linearly Separable (DoS Attack vs Normal)
      self.x_normal = np.random.normal(loc=0.75, scale=0.10, size=150)
      self.y_normal = np.random.normal(loc=0.55, scale=0.15, size=150)
      self.x_attack = np.random.normal(loc=0.25, scale=0.08, size=120)
      self.y_attack = np.random.normal(loc=0.35, scale=0.12, size=120)
      self.target_gini = 0.050
      self.level_title = "Level 1: Linearly Separable (DoS)"
      self.level_desc = "Single-Feature Dominance"

    elif level == 2:
      # Level 2: Non-Linear Sub-Cluster (PortScan / Probe)
      self.x_normal = np.random.normal(loc=0.65, scale=0.12, size=150)
      self.y_normal = np.random.normal(loc=0.60, scale=0.12, size=150)
      self.x_attack = np.random.normal(loc=0.30, scale=0.10, size=120)
      self.y_attack = np.random.normal(loc=0.20, scale=0.08, size=120)
      self.target_gini = 0.120
      self.level_title = "Level 2: Compound Sub-Clusters (Probe)"
      self.level_desc = "Multi-Condition 2D Bounding"

    elif level == 3:
      # Level 3: Overlapping Noise (Advanced Intrusion) — Intentionally Unwinnable (Target <= 0.200)
      self.x_normal = np.random.normal(loc=0.60, scale=0.18, size=150)
      self.y_normal = np.random.normal(loc=0.50, scale=0.18, size=150)
      self.x_attack = np.random.normal(loc=0.40, scale=0.15, size=120)
      self.y_attack = np.random.normal(loc=0.40, scale=0.15, size=120)
      self.target_gini = (
          0.200  # Strict target forces Impurity score to stay Orange/Red (~0.320)
      )
      self.level_title = "Level 3: Overlapping Feature Noise"
      self.level_desc = "Non-Separability & Overfitting Limits"

    # Clip values to continuous feature space [0.05, 0.95]
    self.x_normal = np.clip(self.x_normal, 0.05, 0.95)
    self.y_normal = np.clip(self.y_normal, 0.05, 0.95)
    self.x_attack = np.clip(self.x_attack, 0.05, 0.95)
    self.y_attack = np.clip(self.y_attack, 0.05, 0.95)

  def _show_onboarding_modal(self):
    """Launches the instructional modal dialog on startup."""
    modal = ctk.CTkToplevel(self)
    modal.title("Module Instructions - Decision Tree Splitting")
    modal.geometry("540x480")
    modal.resizable(False, False)
    modal.transient(self)
    modal.grab_set()

    m_frame = ctk.CTkFrame(modal, corner_radius=12, fg_color="#1E293B")
    m_frame.pack(padx=20, pady=20, fill="both", expand=True)

    title = ctk.CTkLabel(
        m_frame,
        text="How to Play: Decision Tree Module",
        font=ctk.CTkFont(size=18, weight="bold"),
        text_color="#FF9800",
    )
    title.pack(pady=(15, 10))

    instructions = (
        "1. OBJECTIVE:\n"
        "   Isolate Normal Traffic (Green) from Attack Traffic (Red) by\n"
        "   adjusting the Vertical (X) and Horizontal (Y) boundary sliders.\n\n"
        "2. 2D QUADRANT PARTITIONING:\n"
        "   Gini Impurity is measured across all 4 quadrants created by\n"
        "   the intersecting X and Y boundary cuts.\n\n"
        "3. TARGET SCORE:\n"
        "   Lower the Weighted Gini score and click 'Complete Level' to unlock\n"
        "   NIDS theoretical takeaways.\n\n"
        "4. PROGRESSION:\n"
        "   Work through Levels 1, 2, and 3 to discover how decision trees\n"
        "   handle linear separation, compound rules, and noise."
    )

    lbl_info = ctk.CTkLabel(
        m_frame,
        text=instructions,
        font=ctk.CTkFont(size=12),
        text_color="#E5E7EB",
        justify="left",
        anchor="w",
        wraplength=460,
    )
    lbl_info.pack(padx=20, pady=10, fill="x")

    btn_start = ctk.CTkButton(
        m_frame,
        text="Start Splitting",
        font=ctk.CTkFont(size=14, weight="bold"),
        fg_color="#FF9800",
        hover_color="#E68A00",
        text_color="#000000",
        height=40,
        command=modal.destroy,
    )
    btn_start.pack(pady=(10, 15))

  def _build_sidebar(self):
    """Left Control Sidebar with fixed geometry propagation."""
    self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
    self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    self.sidebar.grid_propagate(False)  # Prevents sidebar resizing during updates

    title = ctk.CTkLabel(
        self.sidebar,
        text="Module 2: Decision Tree",
        font=ctk.CTkFont(size=20, weight="bold"),
        text_color="#FF9800",
        width=290,
        anchor="w",
    )
    title.pack(pady=(15, 2), padx=15)

    subtitle = ctk.CTkLabel(
        self.sidebar,
        text="Axis-Parallel Feature Space Partitioning",
        font=ctk.CTkFont(size=11, slant="italic"),
        text_color="#A0A0A0",
        width=290,
        anchor="w",
    )
    subtitle.pack(pady=(0, 10), padx=15)

    # Level Selector Buttons
    level_frame = ctk.CTkFrame(self.sidebar, width=290)
    level_frame.pack(fill="x", padx=15, pady=5)

    lbl_lvl = ctk.CTkLabel(
        level_frame,
        text="Select Level Challenge:",
        font=ctk.CTkFont(size=12, weight="bold"),
        text_color="#FFFFFF",
    )
    lbl_lvl.pack(anchor="w", padx=10, pady=(5, 2))

    btn_box = ctk.CTkFrame(level_frame, fg_color="transparent")
    btn_box.pack(fill="x", padx=5, pady=(0, 5))

    self.btn_l1 = ctk.CTkButton(
        btn_box,
        text="L1",
        width=75,
        fg_color="#FF9800",
        text_color="#000000",
        command=lambda: self._select_level(1),
    )
    self.btn_l1.pack(side="left", padx=5, expand=True)

    self.btn_l2 = ctk.CTkButton(
        btn_box,
        text="L2",
        width=75,
        fg_color="#374151",
        text_color="#FFFFFF",
        command=lambda: self._select_level(2),
    )
    self.btn_l2.pack(side="left", padx=5, expand=True)

    self.btn_l3 = ctk.CTkButton(
        btn_box,
        text="L3",
        width=75,
        fg_color="#374151",
        text_color="#FFFFFF",
        command=lambda: self._select_level(3),
    )
    self.btn_l3.pack(side="left", padx=5, expand=True)

    # Controls Box
    controls_frame = ctk.CTkFrame(self.sidebar, width=290)
    controls_frame.pack(fill="x", padx=15, pady=8)

    lbl_v = ctk.CTkLabel(
        controls_frame,
        text="Vertical Boundary (X Split):",
        font=ctk.CTkFont(size=12, weight="bold"),
        text_color="#FFFFFF",
    )
    lbl_v.pack(anchor="w", padx=10, pady=(8, 2))

    self.slider_v = ctk.CTkSlider(
        controls_frame,
        from_=0.05,
        to=0.95,
        number_of_steps=90,
        command=self._on_slider_move,
    )
    self.slider_v.set(self.v_split)
    self.slider_v.pack(fill="x", padx=10, pady=(0, 8))

    lbl_h = ctk.CTkLabel(
        controls_frame,
        text="Horizontal Boundary (Y Split):",
        font=ctk.CTkFont(size=12, weight="bold"),
        text_color="#FFFFFF",
    )
    lbl_h.pack(anchor="w", padx=10, pady=(4, 2))

    self.slider_h = ctk.CTkSlider(
        controls_frame,
        from_=0.05,
        to=0.95,
        number_of_steps=90,
        command=self._on_slider_move,
    )
    self.slider_h.set(self.h_split)
    self.slider_h.pack(fill="x", padx=10, pady=(0, 10))

    # Telemetry Card with Fixed Dimensions
    self.telemetry_card = ctk.CTkFrame(
        self.sidebar,
        width=290,
        height=75,
        fg_color="#1E293B",
        border_width=1,
        border_color="#00E5FF",
    )
    self.telemetry_card.pack(fill="x", padx=15, pady=6)
    self.telemetry_card.pack_propagate(False)

    self.lbl_gini = ctk.CTkLabel(
        self.telemetry_card,
        text="2D Weighted Gini Impurity:\n0.00",
        font=ctk.CTkFont(size=12, weight="bold"),
        text_color="#00E5FF",
        width=280,
    )
    self.lbl_gini.pack(expand=True, fill="both", pady=4)

    # Complete Level Button
    self.btn_complete = ctk.CTkButton(
        self.sidebar,
        text="✓ Complete Level",
        font=ctk.CTkFont(size=13, weight="bold"),
        fg_color="#10B981",
        hover_color="#059669",
        text_color="#FFFFFF",
        height=36,
        command=self._on_complete_level,
    )
    self.btn_complete.pack(fill="x", padx=15, pady=6)

    # Dynamic Tree Visualizer Card with explicit width
    tree_card = ctk.CTkFrame(self.sidebar, width=290, fg_color="#111827")
    tree_card.pack(fill="both", expand=True, padx=15, pady=(0, 10))

    lbl_tree_header = ctk.CTkLabel(
        tree_card,
        text="Generated NIDS Rule Tree:",
        font=ctk.CTkFont(size=12, weight="bold"),
        text_color="#FF9800",
        anchor="w",
    )
    lbl_tree_header.pack(fill="x", padx=10, pady=(6, 2))

    self.lbl_rules = ctk.CTkLabel(
        tree_card,
        text="",
        font=ctk.CTkFont(family="Courier", size=10),
        text_color="#D1D5DB",
        justify="left",
        anchor="nw",
        wraplength=270,
    )
    self.lbl_rules.pack(fill="both", expand=True, padx=10, pady=4)

  def _build_canvas(self):
    """Main Matplotlib Scatter Plot View."""
    self.canvas_frame = ctk.CTkFrame(self)
    self.canvas_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

    plt.style.use("dark_background")
    self.fig, self.ax = plt.subplots(figsize=(7, 6), facecolor="#1A1A1A")
    self.ax.set_facecolor("#121212")

    self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
    self.canvas.get_tk_widget().pack(fill="both", expand=True)

    self._plot_data()
    self._update_telemetry_and_tree()

  def _select_level(self, level):
    """Switches active level, resets sliders to baseline, and redraws."""
    self.btn_l1.configure(
        fg_color="#FF9800" if level == 1 else "#374151",
        text_color="#000000" if level == 1 else "#FFFFFF",
    )
    self.btn_l2.configure(
        fg_color="#FF9800" if level == 2 else "#374151",
        text_color="#000000" if level == 2 else "#FFFFFF",
    )
    self.btn_l3.configure(
        fg_color="#FF9800" if level == 3 else "#374151",
        text_color="#000000" if level == 3 else "#FFFFFF",
    )

    self.v_split = 0.05
    self.h_split = 0.05
    self.slider_v.set(0.05)
    self.slider_h.set(0.05)

    self._load_level_dataset(level)
    self._plot_data()
    self._update_telemetry_and_tree()

  def _plot_data(self):
    """Redraw Scatter Plot Points and User Split Boundaries."""
    self.ax.clear()

    self.ax.scatter(
        self.x_normal,
        self.y_normal,
        c="#00FF66",
        label="Normal Traffic",
        alpha=0.8,
        edgecolors="none",
        s=40,
    )
    self.ax.scatter(
        self.x_attack,
        self.y_attack,
        c="#FF3333",
        label="Attack Traffic",
        alpha=0.8,
        edgecolors="none",
        s=40,
    )

    self.ax.axvline(
        x=self.v_split,
        color="#00E5FF",
        linestyle="--",
        linewidth=2,
        label=f"X Cut ({self.v_split:.2f})",
    )
    self.ax.axhline(
        y=self.h_split,
        color="#FF9800",
        linestyle="--",
        linewidth=2,
        label=f"Y Cut ({self.h_split:.2f})",
    )

    self.ax.set_xlim(0, 1.0)
    self.ax.set_ylim(0, 1.0)
    self.ax.set_title(
        f"{self.level_title} — Target Gini <= {self.target_gini:.3f}",
        color="#FF9800",
        fontsize=12,
        pad=10,
    )
    self.ax.set_xlabel("Feature X: Source Bytes (Normalized)", color="#A0A0A0")
    self.ax.set_ylabel("Feature Y: Packet Count (Normalized)", color="#A0A0A0")
    self.ax.legend(loc="upper right", facecolor="#1F2937", edgecolor="none")
    self.ax.grid(True, color="#2A2A2A", linestyle=":")

    self.fig.tight_layout()
    self.canvas.draw()

  def _on_slider_move(self, val):
    """Callback when user moves split sliders."""
    self.v_split = self.slider_v.get()
    self.h_split = self.slider_h.get()

    self._plot_data()
    self._update_telemetry_and_tree()

  def _calculate_gini(self):
    """Calculates 2D joint weighted Gini Impurity across all 4 quadrants created by X and Y splits."""
    total_pts = len(self.x_normal) + len(self.x_attack)
    weighted_gini = 0.0

    quadrants = [
        (
            (self.x_normal > self.v_split) & (self.y_normal > self.h_split),
            (self.x_attack > self.v_split) & (self.y_attack > self.h_split),
        ),
        (
            (self.x_normal <= self.v_split) & (self.y_normal > self.h_split),
            (self.x_attack <= self.v_split) & (self.y_attack > self.h_split),
        ),
        (
            (self.x_normal <= self.v_split) & (self.y_normal <= self.h_split),
            (self.x_attack <= self.v_split) & (self.y_attack <= self.h_split),
        ),
        (
            (self.x_normal > self.v_split) & (self.y_normal <= self.h_split),
            (self.x_attack > self.v_split) & (self.y_attack <= self.h_split),
        ),
    ]

    for norm_mask, att_mask in quadrants:
      n_norm = np.sum(norm_mask)
      n_att = np.sum(att_mask)
      n_q = n_norm + n_att

      if n_q > 0:
        p_norm = n_norm / n_q
        p_att = n_att / n_q
        gini_q = 1.0 - (p_norm**2 + p_att**2)
        weighted_gini += (n_q / total_pts) * gini_q

    return weighted_gini

  def _update_telemetry_and_tree(self):
    """Updates sidebar text with 2D Gini score and conditional rule tree."""
    gini = self._calculate_gini()

    if gini <= self.target_gini:
      rating = f"(Target Met! <= {self.target_gini:.3f})"
      color = "#00FF66"
    elif gini < 0.38:
      rating = f"(Moderate Impurity | Target: <= {self.target_gini:.3f})"
      color = "#FF9800"
    else:
      rating = f"(High Impurity | Target: <= {self.target_gini:.3f})"
      color = "#FF3333"

    self.lbl_gini.configure(
        text=f"2D Weighted Gini Impurity:\n{gini:.3f}\n{rating}",
        text_color=color,
    )

    rule_text = (
        f"IF src_bytes <= {self.v_split:.2f}:\n"
        f"  ├── IF packet_count <= {self.h_split:.2f}:\n"
        f"  │     └── [Q3: ATTACK BOUNDED]\n"
        f"  └── ELSE (packet_count > {self.h_split:.2f}):\n"
        f"        └── [Q2: MIXED / LOW DENSITY]\n"
        f"ELSE (src_bytes > {self.v_split:.2f}):\n"
        f"  ├── IF packet_count > {self.h_split:.2f}:\n"
        f"  │     └── [Q1: NORMAL TRAFFIC]\n"
        f"  └── ELSE:\n"
        f"        └── [Q4: ISOLATED TRAFFIC]"
    )
    self.lbl_rules.configure(text=rule_text)

  def _on_complete_level(self):
    """Evaluates current solution and displays post-game pedagogical modal."""
    current_gini = self._calculate_gini()

    modal = ctk.CTkToplevel(self)
    modal.title(f"Level {self.current_level} Completion Assessment")
    modal.geometry("580x520")
    modal.resizable(False, False)
    modal.transient(self)
    modal.grab_set()

    # Explicit modal frame container with high contrast background
    m_frame = ctk.CTkFrame(modal, corner_radius=12, fg_color="#1E293B")
    m_frame.pack(padx=20, pady=20, fill="both", expand=True)

    # SUCCESS OR LEVEL 3 ACCEPTANCE CASE
    if current_gini <= self.target_gini or self.current_level == 3:
      if self.current_level == 3:
        title = ctk.CTkLabel(
            m_frame,
            text="⚠️ Level 3 Assessment: Why Can't It Turn Green?",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color="#FF9800",
        )
      else:
        title = ctk.CTkLabel(
            m_frame,
            text=f"🎉 Level {self.current_level} Complete!",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#00FF66",
        )
      title.pack(pady=(15, 5))

      score_lbl = ctk.CTkLabel(
          m_frame,
          text=(
              f"Best Gini Achieved: {current_gini:.3f} (Target:"
              f" <= {self.target_gini:.3f})"
          ),
          font=ctk.CTkFont(size=13, weight="bold"),
          text_color="#00E5FF",
      )
      score_lbl.pack(pady=(0, 10))

      # Post-Game Insights Content
      if self.current_level == 1:
        insight = (
            "WHAT YOU JUST SAW:\n"
            "• All red dots were grouped nicely on the left and green dots on"
            " the right.\n"
            "• You only needed ONE vertical line to separate good traffic from"
            " bad traffic.\n\n"
            "THE BIG TAKEAWAY:\n"
            "• Massive volumetric attacks (like DoS flooding) stick out like a"
            " sore thumb.\n"
            "• A Decision Tree doesn't waste time—it draws a single straight"
            " line (IF packet_size > threshold) and instantly catches the"
            " attack!"
        )
      elif self.current_level == 2:
        insight = (
            "WHAT YOU JUST SAW:\n"
            "• A single vertical line wasn't enough this time—the red attack"
            " dots were hiding down in a specific corner.\n"
            "• You had to use BOTH lines (X and Y) together to trap the red dots"
            " inside their own box.\n\n"
            "THE BIG TAKEAWAY:\n"
            "• Sneaky attacks (like Port Scans) try to blend in by looking"
            " normal on one axis.\n"
            "• Decision Trees solve this by combining rules (IF feature_X is"
            " low AND feature_Y is low). This cuts the graph into smaller boxes"
            " to trap sneaky threats!"
        )
      else:
        insight = (
            "WHAT YOU JUST SAW:\n"
            "• The red and green dots are completely mixed together! No matter"
            " where you move the lines, you always catch some green dots with"
            " the red ones.\n\n"
            "THE BIG TAKEAWAY:\n"
            "• Real-world network data is messy. Attackers intentionally hide"
            " inside normal traffic patterns.\n"
            "• The Overfitting Trap: If you try to force straight lines to"
            " perfectly separate mixed dots, your rules become overly"
            " complicated and break on new data. Non-separable data requires"
            " extra features or Random Forests!"
        )

      lbl_insight = ctk.CTkLabel(
          m_frame,
          text=insight,
          font=ctk.CTkFont(size=12),
          text_color="#E5E7EB",
          justify="left",
          anchor="w",
          wraplength=500,
      )
      lbl_insight.pack(padx=20, pady=10, fill="x")

      # Action Button
      if self.current_level < 3:
        next_lvl = self.current_level + 1
        btn_action = ctk.CTkButton(
            m_frame,
            text=f"Proceed to Level {next_lvl}",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#00E5FF",
            hover_color="#00B3CC",
            text_color="#000000",
            height=38,
            command=lambda: [modal.destroy(), self._select_level(next_lvl)],
        )
      else:
        btn_action = ctk.CTkButton(
            m_frame,
            text="Close & Explore",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#10B981",
            hover_color="#059669",
            text_color="#FFFFFF",
            height=38,
            command=modal.destroy,
        )
      btn_action.pack(pady=(10, 15))

    else:
      # RETRY CASE FOR L1 / L2
      title = ctk.CTkLabel(
          m_frame,
          text="Target Not Met Yet",
          font=ctk.CTkFont(size=18, weight="bold"),
          text_color="#FF9800",
      )
      title.pack(pady=(20, 5))

      msg = (
          f"Current Gini Score: {current_gini:.3f}\n"
          f"Required Target: <= {self.target_gini:.3f}\n\n"
          "Keep adjusting the Vertical (X) and Horizontal (Y) sliders to\n"
          "better isolate the green and red traffic clusters across quadrants!"
      )
      lbl_msg = ctk.CTkLabel(
          m_frame,
          text=msg,
          font=ctk.CTkFont(size=13),
          text_color="#E5E7EB",
          justify="center",
          wraplength=500,
      )
      lbl_msg.pack(padx=20, pady=20)

      btn_retry = ctk.CTkButton(
          m_frame,
          text="Keep Tweaking Sliders",
          font=ctk.CTkFont(size=14, weight="bold"),
          fg_color="#FF9800",
          hover_color="#E68A00",
          text_color="#000000",
          height=38,
          command=modal.destroy,
      )
      btn_retry.pack(pady=(10, 15))


def run_decision_tree_app(parent=None):
  app = DecisionTreeModule(parent)
  app.focus()


if __name__ == "__main__":
  root = ctk.CTk()
  root.withdraw()
  app = DecisionTreeModule()
  app.protocol("WM_DELETE_WINDOW", root.destroy)
  root.mainloop()