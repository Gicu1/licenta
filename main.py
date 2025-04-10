# main.py
import sys
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, scrolledtext
from PIL import Image, ImageTk, ImageSequence
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import shutil
import time
from typing import List, Tuple, Dict, Optional, Any

# Import the refined Robot class
from Robot import Robot, Grid, State, Path

# Define types for configuration dictionariesz
RobotConfig = Dict[str, Any]

# ---------------------- Simulation Core Logic ----------------------

def run_mapf_simulation(grid: Grid,
                        robot_configs: List[RobotConfig],
                        max_steps: int = 200,
                        output_dir: str = "results",
                        temp_frame_dir: str = "temp_frames",
                        log_filename: str = "simulation_log.txt",
                        update_callback: Optional[callable] = None) -> Dict[str, Any]:
    """
    Runs the Multi-Agent Path Finding simulation.

    Args:
        grid (Grid): The environment grid (0=free, 1=obstacle).
        robot_configs (List[RobotConfig]): List of dictionaries, each defining a robot's parameters.
        max_steps (int): Maximum number of simulation steps.
        output_dir (str): Directory to save final results (log, GIF).
        temp_frame_dir (str): Temporary directory to store frames for GIF generation.
        log_filename (str): Name for the simulation log file.
        update_callback (Optional[callable]): A function to call periodically with status updates
                                               (e.g., for GUI feedback). Takes a string message.

    Returns:
        Dict[str, Any]: A dictionary containing simulation results:
            - success (bool): True if the function ran without critical errors.
            - message (str): Status message (e.g., "Simulation complete", error details).
            - gif_path (Optional[str]): Path to the generated GIF, or None if failed.
            - log_path (Optional[str]): Path to the simulation log, or None if failed.
            - final_robots (List[Robot]): The list of Robot objects in their final state.
            - total_steps (int): The number of steps the simulation actually ran.
    """
    results = {
        "success": False,
        "message": "Simulation did not start.",
        "gif_path": None,
        "log_path": None,
        "final_robots": [],
        "total_steps": 0
    }

    if update_callback: update_callback("Preparing simulation environment...")

    # --- Environment Setup ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Clean up temp directory if it exists, then create it
        if os.path.exists(temp_frame_dir):
            shutil.rmtree(temp_frame_dir)
        os.makedirs(temp_frame_dir, exist_ok=True)
    except OSError as e:
        results["message"] = f"Error creating directories: {e}"
        return results

    log_path = os.path.join(output_dir, log_filename)
    gif_path = os.path.join(output_dir, "robot_simulation.gif")
    results["log_path"] = log_path
    results["gif_path"] = gif_path

    # --- Robot Initialization ---
    # CRITICAL: Clear shared state before creating new robots for this run
    Robot.clear_reservations()
    robots: List[Robot] = []
    for i, cfg in enumerate(robot_configs):
        try:
            rob = Robot(
                robot_id=i + 1, # Assign simple numeric ID
                start_x=cfg['start_x'], start_y=cfg['start_y'],
                goal_x=cfg['goal_x'], goal_y=cfg['goal_y'],
                start_time=cfg.get('start_time', 0),
                deadline=cfg.get('deadline', None)
            )
            robots.append(rob)
        except Exception as e:
            results["message"] = f"Error initializing Robot {i+1}: {e}"
            return results
    results["final_robots"] = robots # Store initial state in case loop fails

    if not robots:
        results["message"] = "No robots configured for simulation."
        return results

    # --- Simulation Loop ---
    images = []
    actual_steps = 0
    all_done = False

    try:
        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write("=== Simulation Start ===\n")
            log_file.write(f"Grid Shape: {grid.shape}\n")
            log_file.write(f"Max Steps: {max_steps}\n")
            log_file.write(f"Robots Initial Config: {robot_configs}\n")
            log_file.write("========================\n\n")

            for step in range(max_steps):
                actual_steps = step
                if update_callback: update_callback(f"Running step {step+1}/{max_steps}...")

                log_file.write(f"--- Step {step} ---\n")

                # Determine active robots and sort by priority for this step
                active_robots = [
                    r for r in robots if step >= r.start_time and not r.at_goal() and not r.deadline_missed
                ]
                # Sort: Lower priority value = higher priority = process first
                active_robots.sort(key=lambda r: r.get_priority(step))

                # Log planning order (optional but useful for debugging)
                # planning_order = [r.robot_id for r in active_robots]
                # log_file.write(f"Planning order: {planning_order}\n")

                # Execute step for each active robot IN PRIORITY ORDER
                for r in active_robots:
                    try:
                         r.step(grid, robots, step)
                    except Exception as e:
                         # Catch errors during a specific robot's step
                         print(f"ERROR during step() for Robot {r.robot_id} at step {step}: {e}", file=sys.stderr)
                         log_file.write(f"ERROR during step() for Robot {r.robot_id}: {e}\n")
                         # Optionally mark the robot as failed or continue simulation
                         r.deadline_missed = True # Mark as failed

                # Log robot states after all moves in the step are attempted
                log_file.write("States after step:\n")
                for r_log in robots:
                     log_file.write(f"  {r_log}\n") # Use __repr__

                # --- Visualization Frame Generation ---
                plt.figure(figsize=(8, 8))
                plt.imshow(grid, cmap='binary', origin='lower', interpolation='nearest') # origin='lower' matches array indices to plot coords
                colors = plt.cm.get_cmap('tab10', len(robots)) # Use a colormap

                for i, r in enumerate(robots):
                    marker = 'o'
                    msize = 8
                    final_color = colors(i)
                    if r.at_goal():
                        marker = '*' # Goal reached marker
                        msize = 12
                    elif r.deadline_missed:
                        marker = 'x' # Failed marker
                        msize = 10
                        final_color = 'gray'

                    # Plot current position (y=col, x=row -> plot(col, row))
                    plt.plot(r.y, r.x, marker=marker, color=final_color, markersize=msize, label=f'R{r.robot_id}' if step==0 else "")

                    # Optionally add start/goal markers (can get cluttered)
                    if step == 0:
                         plt.plot(r.start_y, r.start_x, marker='s', color=colors(i), markersize=6, alpha=0.6) # Start square
                         plt.plot(r.goal_y, r.goal_x, marker='D', color=colors(i), markersize=6, alpha=0.6) # Goal diamond

                    # Label text (optional)
                    # label_text = f"R{r.robot_id}"
                    # plt.text(r.y, r.x + 0.1, label_text, color="black", fontsize=8, ha="center", va="bottom")

                plt.title(f"Step {step}")
                plt.xlabel("Column")
                plt.ylabel("Row")
                plt.xticks(np.arange(grid.shape[1]))
                plt.yticks(np.arange(grid.shape[0]))
                plt.grid(True, which='both', color='gray', linewidth=0.5, linestyle=':')
                plt.gca().invert_yaxis() # Make (0,0) top-left

                # Add legend only once
                # if step == 0: plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

                frame_path = os.path.join(temp_frame_dir, f"frame_{step:04d}.png")
                plt.savefig(frame_path, bbox_inches='tight')
                plt.close()

                images.append(Image.open(frame_path))

                # --- Check Termination Condition ---
                if all(r.at_goal() or r.deadline_missed or step < r.start_time for r in robots):
                    log_file.write(f"\n--- Simulation End Condition Met at Step {step} ---\n")
                    all_done = True
                    break

            # End of loop
            if not all_done:
                log_file.write(f"\n--- Simulation Ended: Max Steps ({max_steps}) Reached ---\n")

            results["total_steps"] = actual_steps + 1 # Steps are 0 to actual_steps

    except IOError as e:
         results["message"] = f"Error writing log file: {e}"
         # Continue to try and build GIF if frames exist
    except Exception as e:
         results["message"] = f"An unexpected error occurred during simulation loop: {e}"
         # Try to build GIF with frames generated so far
         print(f"ERROR during simulation: {e}", file=sys.stderr)


    # --- GIF Generation ---
    if images:
        if update_callback: update_callback("Generating GIF animation...")
        try:
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                optimize=False, # Optimization can sometimes be slow/buggy
                duration=500,  # milliseconds per frame
                loop=0         # loop forever
            )
            results["success"] = True # Mark overall success if GIF is created
            results["message"] = f"Simulation complete ({results['total_steps']} steps). Results in {output_dir}"
            results["gif_path"] = gif_path # Confirm path
        except Exception as e:
            results["message"] = f"Simulation ran, but failed to save GIF: {e}"
            results["gif_path"] = None
            print(f"ERROR saving GIF: {e}", file=sys.stderr)
    else:
        results["message"] = "Simulation finished, but no image frames were generated."
        results["gif_path"] = None

    # --- Cleanup ---
    try:
        if os.path.exists(temp_frame_dir):
            shutil.rmtree(temp_frame_dir)
    except OSError as e:
        print(f"Warning: Could not remove temporary frame directory {temp_frame_dir}: {e}", file=sys.stderr)

    return results


# ---------------------- GUI Class ----------------------

class SimulationGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MAPF Simulator")
        self.root.geometry("1000x750") # Adjusted default size

        style = ttk.Style()
        style.theme_use('clam') # Or 'alt', 'default', 'classic'

        # --- Data Storage ---
        self.grid_size: Tuple[int, int] = (20, 20)  # (height, width) -> Consistent with numpy
        self.robots_cfg: List[RobotConfig] = []     # List of robot config dictionaries
        self.obstacles: List[Tuple[int, int]] = []  # List of (row, col) tuples
        self.current_robot_interactive_config: Optional[RobotConfig] = None # For interactive add
        self.edit_robot_list_index: Optional[str] = None # Store Treeview iid of robot being edited

        # GIF animation attributes
        self._gif_frames: List[ImageTk.PhotoImage] = []
        self._gif_job: Optional[str] = None # To store after() job ID for cancellation

        # --- Main Layout ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top: Control Buttons & Status
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.run_button = ttk.Button(control_frame, text="Run Simulation", command=self.run_simulation_gui_callback)
        self.run_button.pack(side=tk.LEFT, padx=(0, 10))
        self.load_defaults_button = ttk.Button(control_frame, text="Load Default Setup", command=self.load_default_setup)
        self.load_defaults_button.pack(side=tk.LEFT, padx=(0, 10))

        self.status_label = ttk.Label(control_frame, text="Status: Ready", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)


        # Main Area: Notebook for configuration and visualization
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.map_tab = ttk.Frame(self.notebook, padding="5")
        self.robot_tab = ttk.Frame(self.notebook, padding="5")
        self.results_tab = ttk.Frame(self.notebook, padding="5") # Renamed from control_tab

        self.notebook.add(self.map_tab, text=" Map Configuration ")
        self.notebook.add(self.robot_tab, text=" Robot Configuration ")
        self.notebook.add(self.results_tab, text=" Simulation Results ")

        # Populate tabs
        self._create_map_tab()
        self._create_robot_tab()
        self._create_results_tab()

        # Initial draw
        self.update_grid_display()
        self.update_robot_list_display()


    # ---------------------- MAP TAB Creation ----------------------
    def _create_map_tab(self):
        """Creates widgets for the Map Configuration tab."""
        # Top frame for size controls
        size_frame = ttk.LabelFrame(self.map_tab, text="Grid Size", padding=10)
        size_frame.pack(padx=5, pady=5, fill=tk.X)

        ttk.Label(size_frame, text="Height (Rows):").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.height_entry = ttk.Spinbox(size_frame, from_=5, to=50, width=5)
        self.height_entry.grid(row=0, column=1, padx=5, pady=2)
        self.height_entry.set(self.grid_size[0]) # height = rows

        ttk.Label(size_frame, text="Width (Cols):").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        self.width_entry = ttk.Spinbox(size_frame, from_=5, to=50, width=5)
        self.width_entry.grid(row=0, column=3, padx=5, pady=2)
        self.width_entry.set(self.grid_size[1]) # width = cols

        ttk.Button(size_frame, text="Apply Size", command=self.update_grid_size).grid(row=0, column=4, padx=10, pady=2)

        # Main area for canvas
        canvas_frame = ttk.Frame(self.map_tab)
        canvas_frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg='white', relief=tk.SUNKEN, borderwidth=1)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Bind default action
        self.canvas.bind("<Button-1>", self._map_canvas_click)

        # Scrollbars (optional, useful for large grids if canvas size is fixed)
        # vsb = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        # hsb = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        # self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        # vsb.pack(side=tk.RIGHT, fill=tk.Y)
        # hsb.pack(side=tk.BOTTOM, fill=tk.X)

        # Bottom frame for instructions and buttons
        bottom_frame = ttk.Frame(self.map_tab)
        bottom_frame.pack(fill=tk.X, pady=5)

        self.map_instruction_label = ttk.Label(bottom_frame, text="Left-click: Toggle Obstacle", foreground="gray", anchor=tk.W)
        self.map_instruction_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        ttk.Button(bottom_frame, text="Add Robot", command=self._start_interactive_add_robot).pack(side=tk.RIGHT, padx=2)
        ttk.Button(bottom_frame, text="Delete Robot", command=self._start_interactive_delete_robot).pack(side=tk.RIGHT, padx=2)


    # ---------------------- ROBOT TAB Creation ----------------------
    def _create_robot_tab(self):
        """Creates widgets for the Robot Configuration tab."""
        # Top frame for manual entry/edit
        config_frame = ttk.LabelFrame(self.robot_tab, text="Add / Edit Robot", padding=10)
        config_frame.pack(padx=5, pady=5, fill=tk.X)

        # Use grid layout for better alignment
        fields = [
            ('start_x', 'Start Row:', 0, 0), ('start_y', 'Start Col:', 0, 2),
            ('goal_x', 'Goal Row:', 1, 0),  ('goal_y', 'Goal Col:', 1, 2),
            ('start_time', 'Start Time:', 2, 0),
            ('deadline', 'Deadline (opt):', 2, 2)
        ]
        self.robot_entries: Dict[str, ttk.Entry] = {}
        for key, label, r, c in fields:
            ttk.Label(config_frame, text=label).grid(row=r, column=c, padx=5, pady=3, sticky=tk.W)
            entry = ttk.Entry(config_frame, width=8)
            entry.grid(row=r, column=c + 1, padx=5, pady=3, sticky=tk.W)
            self.robot_entries[key] = entry
        # Set default values
        self.robot_entries['start_time'].insert(0, "0")

        # Buttons for manual configuration
        button_frame = ttk.Frame(config_frame)
        button_frame.grid(row=3, column=0, columnspan=4, pady=10)

        self.add_manual_button = ttk.Button(button_frame, text="Add Robot", command=self._add_robot_manual)
        self.add_manual_button.pack(side=tk.LEFT, padx=5)
        self.save_edit_button = ttk.Button(button_frame, text="Save Edit", command=self._save_edit_robot, state=tk.DISABLED)
        self.save_edit_button.pack(side=tk.LEFT, padx=5)
        self.cancel_edit_button = ttk.Button(button_frame, text="Cancel Edit", command=self._cancel_edit_robot, state=tk.DISABLED)
        self.cancel_edit_button.pack(side=tk.LEFT, padx=5)

        # Bottom frame for the list of configured robots
        list_frame = ttk.LabelFrame(self.robot_tab, text="Configured Robots", padding=10)
        list_frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        cols = ('start', 'goal', 'start_time', 'deadline')
        self.robot_list_tree = ttk.Treeview(list_frame, columns=cols, show='headings', selectmode='browse')

        self.robot_list_tree.heading('#0', text='ID') # Implicit first column
        # self.robot_list_tree.column('#0', width=40, anchor=tk.CENTER)
        self.robot_list_tree.heading('start', text='Start (R,C)')
        self.robot_list_tree.column('start', width=80, anchor=tk.CENTER)
        self.robot_list_tree.heading('goal', text='Goal (R,C)')
        self.robot_list_tree.column('goal', width=80, anchor=tk.CENTER)
        self.robot_list_tree.heading('start_time', text='Start Time')
        self.robot_list_tree.column('start_time', width=70, anchor=tk.CENTER)
        self.robot_list_tree.heading('deadline', text='Deadline')
        self.robot_list_tree.column('deadline', width=70, anchor=tk.CENTER)


        # Scrollbar for the treeview
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.robot_list_tree.yview)
        self.robot_list_tree.configure(yscrollcommand=scrollbar.set)

        self.robot_list_tree.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        # Bind selection events
        self.robot_list_tree.bind('<<TreeviewSelect>>', self._on_robot_select)
        self.robot_list_tree.bind('<Double-1>', self._edit_robot_from_list) # Double-click to edit
        # Bind delete key
        self.robot_list_tree.bind('<Delete>', self._remove_selected_robot)


    # ---------------------- RESULTS TAB Creation ----------------------
    def _create_results_tab(self):
        """Creates widgets for the Simulation Results tab."""
        # Top part for GIF display
        gif_frame = ttk.LabelFrame(self.results_tab, text="Animation", padding=10)
        gif_frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        # Use a label to display GIF frames
        self.gif_display_label = ttk.Label(gif_frame, text="Run simulation to view animation.", anchor=tk.CENTER)
        self.gif_display_label.pack(fill=tk.BOTH, expand=True)

        # Bottom part for simulation log (optional, can be large)
        log_frame = ttk.LabelFrame(self.results_tab, text="Simulation Log Snippet", padding=10)
        log_frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True, ipady=5) # Add internal padding

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)


    # ---------------------- Grid/Canvas Drawing ----------------------
    def update_grid_display(self):
        """Redraws the map canvas with obstacles and robot markers."""
        self.canvas.delete("all")
        height, width = self.grid_size
        if height <= 0 or width <= 0: return

        # Calculate cell size dynamically based on canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width < 50 or canvas_height < 50: # Wait for canvas to be realized
            self.root.after(50, self.update_grid_display) # Retry after short delay
            return

        cell_w = max(1, canvas_width // width)
        cell_h = max(1, canvas_height // height)
        cell_size = min(cell_w, cell_h, 35) # Use smaller dimension, max size 35px

        grid_pixel_width = width * cell_size
        grid_pixel_height = height * cell_size

        # Center the grid drawing on the canvas
        offset_x = (canvas_width - grid_pixel_width) // 2
        offset_y = (canvas_height - grid_pixel_height) // 2

        self._cell_size = cell_size # Store for click calculations
        self._grid_offset = (offset_x, offset_y)

        # Draw grid lines
        for i in range(height + 1):
            y = offset_y + i * cell_size
            self.canvas.create_line(offset_x, y, offset_x + grid_pixel_width, y, fill="lightgray")
        for j in range(width + 1):
            x = offset_x + j * cell_size
            self.canvas.create_line(x, offset_y, x, offset_y + grid_pixel_height, fill="lightgray")

        # Draw obstacles (filled cells)
        for r, c in self.obstacles:
            if 0 <= r < height and 0 <= c < width:
                x0 = offset_x + c * cell_size
                y0 = offset_y + r * cell_size
                self.canvas.create_rectangle(x0, y0, x0 + cell_size, y0 + cell_size,
                                             fill="black", outline="gray", width=1)

        # Draw robot start/goal markers
        colors = plt.cm.get_cmap('tab10', max(10, len(self.robots_cfg))) # Consistent colors
        for i, robot_cfg in enumerate(self.robots_cfg):
            color = colors(i % 10) # Cycle through first 10 colors
            rgb_color = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'

            s_row, s_col = robot_cfg.get('start_x'), robot_cfg.get('start_y')
            g_row, g_col = robot_cfg.get('goal_x'), robot_cfg.get('goal_y')

            # Draw Start marker (circle)
            if s_row is not None and s_col is not None:
                cx = offset_x + s_col * cell_size + cell_size / 2
                cy = offset_y + s_row * cell_size + cell_size / 2
                rad = cell_size * 0.35
                self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                                        fill=rgb_color, outline="black")
                self.canvas.create_text(cx, cy, text=f"{i+1}", fill="white",
                                        font=("Arial", max(6, int(cell_size * 0.4)), "bold"))

            # Draw Goal marker (square)
            if g_row is not None and g_col is not None:
                off = cell_size * 0.15
                x0 = offset_x + g_col * cell_size + off
                y0 = offset_y + g_row * cell_size + off
                self.canvas.create_rectangle(x0, y0, x0 + cell_size * 0.7, y0 + cell_size * 0.7,
                                             fill=rgb_color, outline="black")
                self.canvas.create_text(x0 + cell_size * 0.35, y0 + cell_size * 0.35, text=f"{i+1}", fill="white",
                                        font=("Arial", max(6, int(cell_size*0.4)), "bold"))

        # Highlight cell being picked interactively
        if self.current_robot_interactive_config:
            s_row = self.current_robot_interactive_config.get('start_x')
            s_col = self.current_robot_interactive_config.get('start_y')
            g_row = self.current_robot_interactive_config.get('goal_x')
            # Highlight start cell if goal not yet picked
            if s_row is not None and s_col is not None and g_row is None:
                x0 = offset_x + s_col * cell_size
                y0 = offset_y + s_row * cell_size
                self.canvas.create_rectangle(x0, y0, x0 + cell_size, y0 + cell_size,
                                            outline="blue", width=3)

    def _get_cell_from_event(self, event: tk.Event) -> Optional[Tuple[int, int]]:
        """Converts canvas click coordinates to grid cell (row, col)."""
        try:
            offset_x, offset_y = self._grid_offset
            cell_size = self._cell_size
            if cell_size <= 0: return None

            col = (event.x - offset_x) // cell_size
            row = (event.y - offset_y) // cell_size

            height, width = self.grid_size
            if 0 <= row < height and 0 <= col < width:
                return row, col
            else:
                return None # Click was outside grid bounds
        except AttributeError: # Handle case where grid hasn't been drawn yet
            return None

    # ---------------------- GUI Callbacks / Actions ----------------------

    def _map_canvas_click(self, event: tk.Event):
        """ Default handler for clicks on the map canvas (toggle obstacle). """
        cell = self._get_cell_from_event(event)
        if cell:
            r, c = cell
            # Check if cell is occupied by a robot start/goal (optional: prevent obstacles there)
            # is_robot_node = any((r,c) == (rcfg.get('start_x'), rcfg.get('start_y')) or \
            #                     (r,c) == (rcfg.get('goal_x'), rcfg.get('goal_y')) for rcfg in self.robots_cfg)
            # if is_robot_node:
            #     messagebox.showwarning("Placement Error", "Cannot place obstacle on a robot's start or goal.")
            #     return

            if cell in self.obstacles:
                self.obstacles.remove(cell)
            else:
                self.obstacles.append(cell)
            self.update_grid_display() # Redraw canvas

    def update_grid_size(self):
        """Applies the new grid size from the entry widgets."""
        try:
            h = int(self.height_entry.get())
            w = int(self.width_entry.get())
            if not (5 <= h <= 50 and 5 <= w <= 50):
                raise ValueError("Grid dimensions must be between 5 and 50.")

            self.grid_size = (h, w)
            # Reset obstacles and robots when grid size changes
            self.obstacles = []
            self.robots_cfg = []
            self.update_robot_list_display()
            self.update_grid_display()
            self._set_status(f"Grid size set to {h}x{w}. Obstacles and robots cleared.")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid grid size: {e}")
            # Reset entries to current valid size
            self.height_entry.set(self.grid_size[0])
            self.width_entry.set(self.grid_size[1])

    def _set_status(self, message: str, color: str = "black"):
        """Updates the status bar label."""
        self.status_label.config(text=f"Status: {message}", foreground=color)
        self.root.update_idletasks() # Force GUI update

    # --- Interactive Robot Add/Delete ---
    def _start_interactive_add_robot(self):
        self._cancel_edit_robot() # Ensure not in manual edit mode
        self.current_robot_interactive_config = {}
        self.notebook.select(self.map_tab)
        self.map_instruction_label.config(text="SELECT START: Click on a free cell for the robot's start position.")
        self.canvas.unbind("<Button-1>") # Unbind default toggle
        self.canvas.bind("<Button-1>", self._get_interactive_robot_start)

    def _get_interactive_robot_start(self, event: tk.Event):
        cell = self._get_cell_from_event(event)
        if cell:
            r, c = cell
            if cell in self.obstacles:
                messagebox.showwarning("Placement Error", "Start position cannot be on an obstacle.")
                return
            # Check if already a start/goal for another robot
            is_node = any((r,c) == (rcfg.get('start_x'), rcfg.get('start_y')) or \
                           (r,c) == (rcfg.get('goal_x'), rcfg.get('goal_y')) for rcfg in self.robots_cfg)
            if is_node:
                 messagebox.showwarning("Placement Error", "Start position cannot be on another robot's start or goal.")
                 return

            self.current_robot_interactive_config['start_x'] = r
            self.current_robot_interactive_config['start_y'] = c
            self.update_grid_display() # Show selection highlight
            self.map_instruction_label.config(text="SELECT GOAL: Click on a different free cell for the goal.")
            self.canvas.unbind("<Button-1>")
            self.canvas.bind("<Button-1>", self._get_interactive_robot_goal)
        else:
            self._cancel_interactive_mode("Invalid click location.")

    def _get_interactive_robot_goal(self, event: tk.Event):
        cell = self._get_cell_from_event(event)
        if cell:
            r, c = cell
            if cell == (self.current_robot_interactive_config.get('start_x'), self.current_robot_interactive_config.get('start_y')):
                messagebox.showwarning("Placement Error", "Goal position cannot be the same as the start position.")
                return
            if cell in self.obstacles:
                messagebox.showwarning("Placement Error", "Goal position cannot be on an obstacle.")
                return
             # Check if already a start/goal for another robot
            is_node = any((r,c) == (rcfg.get('start_x'), rcfg.get('start_y')) or \
                           (r,c) == (rcfg.get('goal_x'), rcfg.get('goal_y')) for rcfg in self.robots_cfg)
            if is_node:
                 messagebox.showwarning("Placement Error", "Goal position cannot be on another robot's start or goal.")
                 return

            self.current_robot_interactive_config['goal_x'] = r
            self.current_robot_interactive_config['goal_y'] = c

            # Get optional parameters (start time, deadline) via dialogs
            start_time = simpledialog.askinteger("Start Time", "Enter robot start time (integer >= 0):",
                                                 parent=self.root, initialvalue=0, minvalue=0)
            if start_time is None: start_time = 0 # Default if cancelled

            deadline_str = simpledialog.askstring("Deadline (Optional)",
                                                   "Enter latest arrival time (integer > start_time), or leave blank for none:",
                                                   parent=self.root)
            deadline: Optional[int] = None
            if deadline_str and deadline_str.strip():
                try:
                    deadline = int(deadline_str)
                    if deadline <= start_time:
                         messagebox.showwarning("Input Error", "Deadline must be greater than start time. Setting to None.")
                         deadline = None
                except ValueError:
                    messagebox.showwarning("Input Error", "Invalid deadline value. Setting to None.")
                    deadline = None

            self.current_robot_interactive_config['start_time'] = start_time
            self.current_robot_interactive_config['deadline'] = deadline

            # Add the configured robot
            self.robots_cfg.append(self.current_robot_interactive_config)
            self.update_robot_list_display()
            self.update_grid_display()
            self._cancel_interactive_mode(f"Robot {len(self.robots_cfg)} added.")
        else:
            self._cancel_interactive_mode("Invalid click location.")

    def _start_interactive_delete_robot(self):
        self._cancel_edit_robot()
        self.current_robot_interactive_config = None # Not adding, just ensure mode is off
        self.notebook.select(self.map_tab)
        self.map_instruction_label.config(text="DELETE ROBOT: Click on a robot's START (circle) or GOAL (square) marker.")
        self.canvas.unbind("<Button-1>")
        self.canvas.bind("<Button-1>", self._get_interactive_robot_delete)

    def _get_interactive_robot_delete(self, event: tk.Event):
        cell = self._get_cell_from_event(event)
        if cell:
            r, c = cell
            found_index = -1
            for i, rob_cfg in enumerate(self.robots_cfg):
                if (r, c) == (rob_cfg.get('start_x'), rob_cfg.get('start_y')) or \
                   (r, c) == (rob_cfg.get('goal_x'), rob_cfg.get('goal_y')):
                    found_index = i
                    break

            if found_index != -1:
                confirm = messagebox.askyesno("Confirm Deletion", f"Delete Robot {found_index + 1}?", parent=self.root)
                if confirm:
                    del self.robots_cfg[found_index]
                    self.update_robot_list_display()
                    self.update_grid_display()
                    self._cancel_interactive_mode(f"Robot {found_index + 1} deleted.")
                else:
                    self._cancel_interactive_mode("Deletion cancelled.")
            else:
                # Don't cancel mode, allow user to try clicking again
                 self.map_instruction_label.config(text="DELETE ROBOT: No robot found at that cell. Click a START or GOAL marker.")
        else:
            self._cancel_interactive_mode("Invalid click location.")


    def _cancel_interactive_mode(self, status_message: str = "Interactive mode cancelled."):
        """Resets canvas binding and instructions after interactive mode."""
        self.canvas.unbind("<Button-1>")
        self.canvas.bind("<Button-1>", self._map_canvas_click) # Rebind default
        self.map_instruction_label.config(text="Left-click: Toggle Obstacle")
        self.current_robot_interactive_config = None
        self.update_grid_display() # Remove any highlights
        self._set_status(status_message)


    # --- Manual Robot Config ---
    def _clear_robot_entries(self):
        """Clears the text in the robot configuration entry fields."""
        for key, entry in self.robot_entries.items():
            is_disabled = (entry.cget('state') == tk.DISABLED)
            if not is_disabled: entry.config(state=tk.NORMAL)
            entry.delete(0, tk.END)
            if key == 'start_time': entry.insert(0, "0") # Reset default


    def _validate_robot_config_inputs(self) -> Optional[RobotConfig]:
        """Reads and validates inputs from the manual entry fields."""
        config = {}
        try:
            h, w = self.grid_size
            config['start_x'] = int(self.robot_entries['start_x'].get())
            config['start_y'] = int(self.robot_entries['start_y'].get())
            config['goal_x'] = int(self.robot_entries['goal_x'].get())
            config['goal_y'] = int(self.robot_entries['goal_y'].get())
            config['start_time'] = int(self.robot_entries['start_time'].get())
            deadline_str = self.robot_entries['deadline'].get().strip()
            config['deadline'] = int(deadline_str) if deadline_str else None

            # Basic validation
            if not (0 <= config['start_x'] < h and 0 <= config['start_y'] < w):
                raise ValueError("Start position is out of grid bounds.")
            if not (0 <= config['goal_x'] < h and 0 <= config['goal_y'] < w):
                raise ValueError("Goal position is out of grid bounds.")
            if (config['start_x'], config['start_y']) == (config['goal_x'], config['goal_y']):
                raise ValueError("Start and Goal positions cannot be the same.")
            if config['start_time'] < 0:
                raise ValueError("Start time must be non-negative.")
            if config['deadline'] is not None and config['deadline'] <= config['start_time']:
                raise ValueError("Deadline must be greater than start time.")
            # Check for overlaps with obstacles
            if (config['start_x'], config['start_y']) in self.obstacles:
                 raise ValueError("Start position cannot be on an obstacle.")
            if (config['goal_x'], config['goal_y']) in self.obstacles:
                 raise ValueError("Goal position cannot be on an obstacle.")

            # Check for overlaps with existing robot start/goals (allow editing same robot)
            editing_idx = int(self.edit_robot_list_index) if self.edit_robot_list_index else -1
            for i, rcfg in enumerate(self.robots_cfg):
                 if i == editing_idx: continue # Skip check if editing this robot
                 if (config['start_x'], config['start_y']) == (rcfg.get('start_x'), rcfg.get('start_y')) or \
                    (config['start_x'], config['start_y']) == (rcfg.get('goal_x'), rcfg.get('goal_y')):
                      raise ValueError(f"Start position conflicts with Robot {i+1}'s start/goal.")
                 if (config['goal_x'], config['goal_y']) == (rcfg.get('start_x'), rcfg.get('start_y')) or \
                    (config['goal_x'], config['goal_y']) == (rcfg.get('goal_x'), rcfg.get('goal_y')):
                      raise ValueError(f"Goal position conflicts with Robot {i+1}'s start/goal.")

            return config

        except ValueError as e:
            messagebox.showerror("Input Error", str(e), parent=self.root)
            return None
        except Exception as e: # Catch other potential errors like non-integer input
             messagebox.showerror("Input Error", f"Invalid input: {e}", parent=self.root)
             return None

    def _add_robot_manual(self):
        """Adds a robot using the values from the manual entry fields."""
        config = self._validate_robot_config_inputs()
        if config:
            self.robots_cfg.append(config)
            self.update_robot_list_display()
            self.update_grid_display()
            self._clear_robot_entries()
            self._set_status(f"Robot {len(self.robots_cfg)} added manually.")

    def _on_robot_select(self, event: tk.Event):
        """Handles selection change in the robot list Treeview."""
        selected_items = self.robot_list_tree.selection()
        if not selected_items:
            # Selection cleared, potentially disable edit button if not already editing
            if self.edit_robot_list_index is None:
                 self._cancel_edit_robot() # Ensure edit mode is fully off
            return

        # If not currently editing, allow selection to potentially start an edit
        if self.edit_robot_list_index is None:
             pass # Just selecting, don't load into fields automatically unless double-clicked


    def _edit_robot_from_list(self, event: tk.Event):
        """Starts editing the double-clicked robot from the list."""
        selected_items = self.robot_list_tree.selection()
        if not selected_items: return
        item_id = selected_items[0] # Treeview uses generated IDs (like I001) or integer IDs
        try:
            # Find the index in our actual list corresponding to the selected treeview item
            # Assuming item IDs are string representations of list indices '0', '1', etc.
            robot_index = int(item_id)
            if 0 <= robot_index < len(self.robots_cfg):
                self._start_edit_robot(robot_index, item_id)
            else:
                 print(f"WARN: Treeview item ID '{item_id}' out of range for robots_cfg list.", file=sys.stderr)

        except ValueError:
             print(f"WARN: Could not parse robot index from Treeview item ID '{item_id}'.", file=sys.stderr)


    def _start_edit_robot(self, robot_index: int, item_id: str):
        """Loads robot data into entry fields and enables edit mode."""
        self._cancel_interactive_mode("Manual edit started.") # Exit interactive modes
        self.edit_robot_list_index = item_id # Store the Treeview Item ID being edited
        robot_cfg = self.robots_cfg[robot_index]

        self._clear_robot_entries()
        self.robot_entries['start_x'].insert(0, str(robot_cfg.get('start_x', '')))
        self.robot_entries['start_y'].insert(0, str(robot_cfg.get('start_y', '')))
        self.robot_entries['goal_x'].insert(0, str(robot_cfg.get('goal_x', '')))
        self.robot_entries['goal_y'].insert(0, str(robot_cfg.get('goal_y', '')))
        self.robot_entries['start_time'].insert(0, str(robot_cfg.get('start_time', '0')))
        deadline = robot_cfg.get('deadline')
        self.robot_entries['deadline'].insert(0, str(deadline) if deadline is not None else "")

        self.save_edit_button.config(state=tk.NORMAL)
        self.cancel_edit_button.config(state=tk.NORMAL)
        self.add_manual_button.config(state=tk.DISABLED)
        self.robot_list_tree.config(selectmode='none') # Prevent selecting others during edit

    def _save_edit_robot(self):
        """Saves the edited robot configuration."""
        if self.edit_robot_list_index is None: return # Should not happen if buttons are managed correctly

        # Validate the current entries
        config = self._validate_robot_config_inputs()
        if config:
            try:
                # Find the actual list index from the stored Treeview ID
                 robot_index = int(self.edit_robot_list_index)
                 if 0 <= robot_index < len(self.robots_cfg):
                     # Update the configuration in the list
                     self.robots_cfg[robot_index] = config
                     self.update_robot_list_display()
                     self.update_grid_display()
                     self._cancel_edit_robot() # Clear fields, reset buttons
                     self._set_status(f"Robot {robot_index + 1} updated.")
                 else:
                      messagebox.showerror("Error", "Failed to save edit: Robot index out of sync.", parent=self.root)
                      self._cancel_edit_robot()
            except ValueError:
                 messagebox.showerror("Error", "Failed to save edit: Invalid robot identifier.", parent=self.root)
                 self._cancel_edit_robot()


    def _cancel_edit_robot(self):
        """Cancels the current edit operation."""
        self.edit_robot_list_index = None
        self._clear_robot_entries()
        self.save_edit_button.config(state=tk.DISABLED)
        self.cancel_edit_button.config(state=tk.DISABLED)
        self.add_manual_button.config(state=tk.NORMAL)
        self.robot_list_tree.config(selectmode='browse') # Re-enable selection

    def _remove_selected_robot(self, event: Optional[tk.Event] = None):
        """Removes the robot currently selected in the Treeview."""
        if self.edit_robot_list_index is not None:
             messagebox.showwarning("Action Blocked", "Cannot remove robot while in edit mode. Cancel or save edit first.", parent=self.root)
             return

        selected_items = self.robot_list_tree.selection()
        if not selected_items:
            messagebox.showwarning("Selection Error", "Select a robot from the list to remove.", parent=self.root)
            return

        item_id = selected_items[0]
        try:
             robot_index = int(item_id)
             if 0 <= robot_index < len(self.robots_cfg):
                confirm = messagebox.askyesno("Confirm Deletion", f"Delete Robot {robot_index + 1}?", parent=self.root)
                if confirm:
                    del self.robots_cfg[robot_index]
                    self.update_robot_list_display()
                    self.update_grid_display()
                    self._set_status(f"Robot {robot_index + 1} removed.")
             else:
                  print(f"WARN: Treeview item ID '{item_id}' out of range for removal.", file=sys.stderr)
        except ValueError:
             print(f"WARN: Could not parse robot index for removal from Treeview ID '{item_id}'.", file=sys.stderr)


    def update_robot_list_display(self):
        """Updates the Treeview list with current robot configurations."""
        # Clear existing items
        for item in self.robot_list_tree.get_children():
            self.robot_list_tree.delete(item)
        # Add current robots
        for i, r_cfg in enumerate(self.robots_cfg):
            start_str = f"({r_cfg.get('start_x', '?')},{r_cfg.get('start_y', '?')})"
            goal_str = f"({r_cfg.get('goal_x', '?')},{r_cfg.get('goal_y', '?')})"
            start_time = r_cfg.get('start_time', 0)
            deadline = r_cfg.get('deadline', 'None')
            # Use 'i' as the item ID (iid), assuming list order matches display order
            self.robot_list_tree.insert('', tk.END, iid=str(i),
                                        values=(start_str, goal_str, start_time, deadline))

    # ---------------------- Simulation Control ----------------------

    def run_simulation_gui_callback(self):
        """Handles the 'Run Simulation' button click."""
        self._cancel_edit_robot() # Ensure not editing
        self._cancel_interactive_mode("Starting simulation...") # Ensure not in interactive mode

        # Disable run button during simulation
        self.run_button.config(state=tk.DISABLED)
        self.load_defaults_button.config(state=tk.DISABLED) # Also disable loading defaults

        # Clear previous results display
        self._clear_results_display()

        # 1. Prepare Grid
        h, w = self.grid_size
        grid_np = np.zeros((h, w), dtype=int)
        for r, c in self.obstacles:
            if 0 <= r < h and 0 <= c < w:
                grid_np[r, c] = 1

        # 2. Prepare Robot Configs (already stored in self.robots_cfg)
        if not self.robots_cfg:
             messagebox.showerror("Error", "No robots configured. Add robots before running.")
             self.run_button.config(state=tk.NORMAL)
             self.load_defaults_button.config(state=tk.NORMAL)
             return

        # 3. Define output paths
        output_dir = os.path.join(os.getcwd(), "simulation_results")
        temp_dir = os.path.join(output_dir, "temp_frames")

        # 4. Define update callback for GUI feedback
        def gui_update_callback(message: str):
            self._set_status(message, "blue")

        # 5. Run the simulation (consider running in a separate thread for long sims)
        # For simplicity here, run directly in the main thread (GUI will freeze)
        self._set_status("Simulation running...", "blue")
        start_time = time.time()

        sim_results = run_mapf_simulation(
            grid=grid_np,
            robot_configs=self.robots_cfg,
            max_steps=300, # Increased max steps
            output_dir=output_dir,
            temp_frame_dir=temp_dir,
            update_callback=gui_update_callback
        )

        end_time = time.time()
        sim_duration = end_time - start_time
        print(f"Simulation and GIF generation took {sim_duration:.2f} seconds.")

        # 6. Process Results
        self._set_status(sim_results["message"], "green" if sim_results["success"] else "red")

        if sim_results["gif_path"]:
            self._display_gif(sim_results["gif_path"])
            self.notebook.select(self.results_tab) # Switch to results tab

        if sim_results["log_path"]:
             self._display_log_snippet(sim_results["log_path"])

        # Re-enable button
        self.run_button.config(state=tk.NORMAL)
        self.load_defaults_button.config(state=tk.NORMAL)


    def _clear_results_display(self):
        """Clears the GIF and log display areas."""
        # Stop existing animation if running
        if self._gif_job:
            self.root.after_cancel(self._gif_job)
            self._gif_job = None
        self._gif_frames = []
        self.gif_display_label.config(image=None, text="Run simulation to view animation.")

        # Clear log text
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _display_gif(self, path: str):
        """Loads and starts animating the simulation GIF."""
        self._gif_frames = []
        try:
            img = Image.open(path)
            # Resize frames for display (adjust size as needed)
            max_w, max_h = 500, 500
            for frame in ImageSequence.Iterator(img):
                 frame_copy = frame.copy().convert("RGBA") # Ensure correct mode
                 # Calculate aspect ratio respecting resize
                 frame_copy.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                 self._gif_frames.append(ImageTk.PhotoImage(frame_copy))

            if self._gif_frames:
                 self._animate_gif(0)
                 self.gif_display_label.config(text="") # Remove placeholder text
            else:
                 self.gif_display_label.config(text="GIF created, but no frames found.")

        except FileNotFoundError:
             self.gif_display_label.config(text=f"Error: GIF file not found at\n{path}")
        except Exception as e:
            self.gif_display_label.config(text=f"Error loading GIF: {e}")
            print(f"ERROR loading GIF: {e}", file=sys.stderr)

    def _animate_gif(self, frame_index: int):
        """Callback to display the next frame of the GIF."""
        if not self._gif_frames: return # Stop if frames cleared

        frame = self._gif_frames[frame_index]
        self.gif_display_label.config(image=frame)

        next_index = (frame_index + 1) % len(self._gif_frames)
        # Schedule the next frame update (duration matches GIF save duration)
        self._gif_job = self.root.after(500, self._animate_gif, next_index)

    def _display_log_snippet(self, log_path: str, lines: int = 50):
        """Displays the last few lines of the simulation log file."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                # Read lines efficiently (especially for large logs)
                log_lines = f.readlines() # Read all lines
                start_index = max(0, len(log_lines) - lines) # Get last 'lines' lines
                snippet = "".join(log_lines[start_index:])
                self.log_text.insert(tk.END, f"... (last {lines} lines of {log_path}) ...\n\n")
                self.log_text.insert(tk.END, snippet)
        except FileNotFoundError:
            self.log_text.insert(tk.END, f"Log file not found: {log_path}")
        except Exception as e:
            self.log_text.insert(tk.END, f"Error reading log file: {e}")
        self.log_text.config(state=tk.DISABLED) # Make read-only

    # ---------------------- Default Setup ----------------------
    def load_default_setup(self):
        """Loads a predefined grid, obstacles, and robot configuration."""
        self._cancel_edit_robot()
        self._cancel_interactive_mode("Loading default setup...")

        self.grid_size = (20, 20)
        self.height_entry.set(self.grid_size[0])
        self.width_entry.set(self.grid_size[1])


        self.obstacles = [
            (5,4), (5,5), (5,6), (5,7), (15,15), (15,16),
            (2,9), (2,11), (3,9), (3,11), (4,9), (4,11), (5,9), (5,11),
            (6,9), (6,11), (7,9), (7,11), (8,9), (8,11), (10,9), (10,11),
            (12,9), (12,11), (14,9), (14,11), (15,9), (15,11), (16,9), (16,11),
            (8,12), (9,12), (10,12), (8,8), (9,8), (10,8), (5,8),
            (8,13), (8,14), (8,15), (9,9), (9,11),
            (12,8), (14,8), (12,12), (14,12), (13,8), (13,12),
            (8,16), (8,17), (8,18), (8,19), (5,3), (5,2), (5,1),
            (3,0), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7),
            (7,0), (7,1), (7,2), (7,3), (7,4), (7,5), (7,6),
            (9,1), (9,2), (9,3), (9,4), (9,5), (9,6), (9,7),
            (11,0), (11,1), (11,2), (11,3), (11,4), (11,5), (11,6),
            (13,9), (13,11), (11,8), (11,12),        
        ]

        self.robots_cfg = [
            {'start_x': 0,  'start_y':10, 'goal_x':18, 'goal_y':10, 'start_time':0,  'deadline':50},
            {'start_x':18, 'start_y':9,  'goal_x':1,  'goal_y':11, 'start_time':5,  'deadline':60},
            {'start_x':18, 'start_y':0,  'goal_x':0,  'goal_y':18, 'start_time':0,  'deadline':40},
            {'start_x':0,  'start_y':18, 'goal_x':18, 'goal_y':0, 'start_time':10, 'deadline':55},
            {'start_x':10, 'start_y':13, 'goal_x':18, 'goal_y':18, 'start_time':0,  'deadline':35},
            {'start_x':6,  'start_y':12, 'goal_x':6,  'goal_y':12, 'start_time':0,  'deadline':None},
        ]

        self.update_robot_list_display()
        self.update_grid_display()
        self._set_status("Default setup loaded.")


# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    root = tk.Tk()
    gui = SimulationGUI(root)
    # Add binding to redraw canvas on resize (important for dynamic cell size)
    gui.canvas.bind("<Configure>", lambda event: gui.update_grid_display())
    root.mainloop()