import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk, ImageSequence
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import shutil

# Import your updated Robot class
from Robot import Robot

class SimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Path Planning Simulator")
        style = ttk.Style()
        style.theme_use('clam')
        
        # Default simulation parameters
        self.grid_size = (20, 20)  # (width, height)
        self.robots = []     # List of robot dictionaries
        self.obstacles = []  # List of (row, col)
        self.current_robot_config = {}
        self.edit_robot_index = None
        self.gif_frames = []
        
        # Main Notebook
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tabs
        self.map_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.map_tab, text="Map Configuration")
        self.create_map_tab()
        
        self.robot_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.robot_tab, text="Robot Configuration")
        self.create_robot_tab()
        
        self.control_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.control_tab, text="Simulation Control")
        self.create_control_tab()
        
        # Initial draw
        self.update_grid_display()

    # ---------------------- MAP TAB ----------------------
    def create_map_tab(self):
        size_frame = ttk.LabelFrame(self.map_tab, text="Grid Size", padding=10)
        size_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ttk.Label(size_frame, text="Width:").grid(row=0, column=0, padx=5, pady=2)
        self.width_entry = ttk.Spinbox(size_frame, from_=5, to=50, width=5)
        self.width_entry.grid(row=0, column=1, padx=5, pady=2)
        self.width_entry.set(self.grid_size[0])
        
        ttk.Label(size_frame, text="Height:").grid(row=0, column=2, padx=5, pady=2)
        self.height_entry = ttk.Spinbox(size_frame, from_=5, to=50, width=5)
        self.height_entry.grid(row=0, column=3, padx=5, pady=2)
        self.height_entry.set(self.grid_size[1])
        
        ttk.Button(size_frame, text="Apply Size", command=self.update_grid_size).grid(row=0, column=4, padx=10, pady=2)
        
        self.canvas = tk.Canvas(self.map_tab, bg='white', relief=tk.SUNKEN, borderwidth=2)
        self.canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        # Default left-click toggles obstacles
        self.canvas.bind("<Button-1>", self.toggle_obstacle)
        
        self.map_instruction_label = ttk.Label(self.map_tab, text="Click on a cell to toggle an obstacle", foreground="gray")
        self.map_instruction_label.pack(pady=5)
        
        btn_frame = ttk.Frame(self.map_tab)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Add Robot (Interactive)", command=self.start_interactive_add_robot).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Delete Robot (Interactive)", command=self.start_interactive_delete_robot).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(self.map_tab, text="Load Default Setup", command=self.load_default_setup).pack(pady=5)

    # ---------------------- ROBOT TAB ----------------------
    def create_robot_tab(self):
        config_frame = ttk.LabelFrame(self.robot_tab, text="Robot Parameters", padding=10)
        config_frame.pack(padx=10, pady=5, fill=tk.X)
        
        fields = [
            ('start_x', 'Start Row:'), ('start_y', 'Start Col:'),
            ('goal_x', 'Goal Row:'),  ('goal_y', 'Goal Col:'),
            ('speed', 'Speed:'),      ('start_time', 'Start Time:'),
            ('deadline', 'Deadline:')
        ]
        self.entries = {}
        for i, (key, label) in enumerate(fields):
            ttk.Label(config_frame, text=label).grid(row=i//2, column=(i % 2)*2, padx=5, pady=2, sticky=tk.W)
            e = ttk.Entry(config_frame, width=8)
            e.grid(row=i//2, column=(i % 2)*2+1, padx=5, pady=2)
            self.entries[key] = e
        
        btn_frame = ttk.Frame(self.robot_tab)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Add Robot (Manual)", command=self.add_robot).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Edit Robot", command=self.edit_robot).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Remove Robot", command=self.remove_robot).pack(side=tk.LEFT, padx=5)
        self.save_edit_button = ttk.Button(btn_frame, text="Save Edit", command=self.save_edit_robot, state=tk.DISABLED)
        self.save_edit_button.pack(side=tk.LEFT, padx=5)
        self.cancel_edit_button = ttk.Button(btn_frame, text="Cancel Edit", command=self.cancel_edit_robot, state=tk.DISABLED)
        self.cancel_edit_button.pack(side=tk.LEFT, padx=5)
        
        list_frame = ttk.LabelFrame(self.robot_tab, text="Configured Robots", padding=10)
        list_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        self.robot_list = ttk.Treeview(list_frame, columns=('start', 'goal', 'speed', 'start_time', 'deadline'))
        self.robot_list.heading('#0', text='#')
        self.robot_list.column('#0', width=40)
        for col in self.robot_list['columns']:
            self.robot_list.heading(col, text=col.capitalize())
            self.robot_list.column(col, width=80)
            
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.robot_list.yview)
        self.robot_list.configure(yscrollcommand=scrollbar.set)
        self.robot_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # ---------------------- CONTROL TAB ----------------------
    def create_control_tab(self):
        ttk.Button(self.control_tab, text="Run Simulation", command=self.run_simulation).pack(pady=20)
        self.status_label = ttk.Label(self.control_tab, text="", foreground="green", font=("Arial", 10, "italic"))
        self.status_label.pack(pady=5)
        self.gif_label = ttk.Label(self.control_tab)
        self.gif_label.pack(pady=10)
    
    # ---------------------- GRID DISPLAY ----------------------
    def update_grid_size(self):
        try:
            w = int(self.width_entry.get())
            h = int(self.height_entry.get())
            if w < 5 or h < 5 or w > 50 or h > 50:
                raise ValueError
            self.grid_size = (w, h)
            self.obstacles = []
            self.robots = []
            self.update_robot_list()
            self.update_grid_display()
            self.map_instruction_label.config(text="Grid size updated.")
        except ValueError:
            messagebox.showerror("Error", "Invalid grid size (5..50).")
    
    def update_grid_display(self):
        """
        Redraws the canvas with obstacles and robot markers.
        The canvas is `width × height` cells.
        Each cell has size = min(500 // max(width, height), 30).
        """
        self.canvas.delete("all")
        
        width, height = self.grid_size
        cell_size = min(500 // max(width, height), 30)
        
        # Draw cells
        for row in range(height):
            for col in range(width):
                x0 = col * cell_size
                y0 = row * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                color = 'black' if (row, col) in self.obstacles else 'white'
                self.canvas.create_rectangle(x0, y0, x1, y1,
                                             fill=color, outline="gray")
        
        # Draw each robot's start/goal
        for i, robot_cfg in enumerate(self.robots):
            srow = robot_cfg.get('start_x')
            scol = robot_cfg.get('start_y')
            grow = robot_cfg.get('goal_x')
            gcol = robot_cfg.get('goal_y')
            # Draw Start marker (blue)
            if srow is not None and scol is not None:
                sx = scol * cell_size + cell_size/2
                sy = srow * cell_size + cell_size/2
                self.canvas.create_text(sx, sy,
                                        text=f"S{i+1}",
                                        fill="blue",
                                        font=("Arial", max(10, int(cell_size/2)), "bold"))
            # Draw Goal marker (red)
            if grow is not None and gcol is not None:
                gx = gcol * cell_size + cell_size/2
                gy = grow * cell_size + cell_size/2
                self.canvas.create_text(gx, gy,
                                        text=f"G{i+1}",
                                        fill="red",
                                        font=("Arial", max(10, int(cell_size/2)), "bold"))
        
        # If we are in the process of picking start for a robot (start picked, but not goal yet),
        # highlight that cell
        if self.current_robot_config and 'start_x' in self.current_robot_config and 'goal_x' not in self.current_robot_config:
            srow = self.current_robot_config['start_x']
            scol = self.current_robot_config['start_y']
            x0 = scol * cell_size
            y0 = srow * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            self.canvas.create_rectangle(x0, y0, x1, y1,
                                         fill="lightblue", outline="blue")

    def toggle_obstacle(self, event):
        """
        Left-click to add/remove an obstacle.
        """
        width, height = self.grid_size
        cell_size = min(500 // max(width, height), 30)
        grid_col = event.x // cell_size
        grid_row = event.y // cell_size
        
        if 0 <= grid_row < height and 0 <= grid_col < width:
            if (grid_row, grid_col) in self.obstacles:
                self.obstacles.remove((grid_row, grid_col))
            else:
                self.obstacles.append((grid_row, grid_col))
            self.update_grid_display()
    
    # ---------------------- INTERACTIVE ADDITION ----------------------
    def start_interactive_add_robot(self):
        self.current_robot_config = {}
        self.notebook.select(self.map_tab)
        self.map_instruction_label.config(text="Interactive Add: Click on a cell for the START position")
        self.canvas.unbind("<Button-1>")
        self.canvas.bind("<Button-1>", self.get_robot_start)
    
    def get_robot_start(self, event):
        w, h = self.grid_size
        cell_size = min(500 // max(w, h), 30)
        col = event.x // cell_size
        row = event.y // cell_size
        if 0 <= row < h and 0 <= col < w:
            self.current_robot_config['start_x'] = row
            self.current_robot_config['start_y'] = col
            self.update_grid_display()
            self.map_instruction_label.config(text="Now click on a cell for the GOAL position")
            self.canvas.unbind("<Button-1>")
            self.canvas.bind("<Button-1>", self.get_robot_goal)
        else:
            messagebox.showerror("Error", "Start out of bounds.")
            self.cancel_interactive_mode()
    
    def get_robot_goal(self, event):
        w, h = self.grid_size
        cell_size = min(500 // max(w, h), 30)
        col = event.x // cell_size
        row = event.y // cell_size
        if 0 <= row < h and 0 <= col < w:
            self.current_robot_config['goal_x'] = row
            self.current_robot_config['goal_y'] = col
            self.canvas.unbind("<Button-1>")
            # Ask for optional start_time and deadline
            delay = simpledialog.askinteger("Start Time", "Enter robot start time (default 0):", parent=self.root, initialvalue=0)
            if delay is None:
                delay = 0
            self.current_robot_config['start_time'] = delay
            
            deadline_str = simpledialog.askstring("Deadline", "Enter deadline (leave blank for none):", parent=self.root)
            if deadline_str and deadline_str.strip():
                try:
                    deadline = int(deadline_str)
                except ValueError:
                    messagebox.showerror("Error", "Invalid deadline. Using None.")
                    deadline = None
            else:
                deadline = None
            self.current_robot_config['deadline'] = deadline
            self.current_robot_config['speed'] = 1  # default
            
            # Save
            self.robots.append(self.current_robot_config)
            self.update_robot_list()
            self.update_grid_display()
            self.map_instruction_label.config(text="Robot added.")
            self.canvas.bind("<Button-1>", self.toggle_obstacle)
        else:
            messagebox.showerror("Error", "Goal out of bounds.")
            self.cancel_interactive_mode()
    
    def cancel_interactive_mode(self):
        """
        Rebind the default obstacle toggling.
        """
        self.canvas.unbind("<Button-1>")
        self.canvas.bind("<Button-1>", self.toggle_obstacle)
        self.current_robot_config = {}
        self.map_instruction_label.config(text="Canceled interactive robot creation.")
        self.update_grid_display()

    # ---------------------- INTERACTIVE DELETION ----------------------
    def start_interactive_delete_robot(self):
        self.notebook.select(self.map_tab)
        self.map_instruction_label.config(text="Interactive Delete: click on a robot's start or goal marker to remove it.")
        self.canvas.unbind("<Button-1>")
        self.canvas.bind("<Button-1>", self.get_robot_to_delete)
    
    def get_robot_to_delete(self, event):
        w, h = self.grid_size
        cell_size = min(500 // max(w, h), 30)
        col = event.x // cell_size
        row = event.y // cell_size
        
        found_index = None
        for i, rob in enumerate(self.robots):
            sx, sy = rob['start_x'], rob['start_y']
            gx, gy = rob['goal_x'], rob['goal_y']
            if (row == sx and col == sy) or (row == gx and col == gy):
                found_index = i
                break
        if found_index is not None:
            confirm = messagebox.askyesno("Confirm", f"Delete robot #{found_index+1}?")
            if confirm:
                del self.robots[found_index]
                self.update_robot_list()
                self.update_grid_display()
                messagebox.showinfo("Deleted", "Robot removed.")
        else:
            messagebox.showwarning("Not Found", "No robot found at that cell.")
        
        # Restore default
        self.canvas.unbind("<Button-1>")
        self.canvas.bind("<Button-1>", self.toggle_obstacle)
        self.map_instruction_label.config(text="Click on a cell to toggle an obstacle.")

    # ---------------------- MANUAL ADDITION ----------------------
    def add_robot(self):
        try:
            config = {
                'start_x': int(self.entries['start_x'].get()),
                'start_y': int(self.entries['start_y'].get()),
                'goal_x': int(self.entries['goal_x'].get()),
                'goal_y': int(self.entries['goal_y'].get()),
                'speed': int(self.entries['speed'].get()),
                'start_time': int(self.entries['start_time'].get()),
                'deadline': int(self.entries['deadline'].get()) if self.entries['deadline'].get().strip() else None
            }
            w, h = self.grid_size
            # Validate in-bounds
            if not (0 <= config['start_x'] < h and 0 <= config['start_y'] < w):
                raise ValueError("Start out of bounds")
            if not (0 <= config['goal_x'] < h and 0 <= config['goal_y'] < w):
                raise ValueError("Goal out of bounds")
            
            self.robots.append(config)
            self.update_robot_list()
            self.update_grid_display()
            messagebox.showinfo("Success", "Robot added.")
            self.clear_entries()
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def edit_robot(self):
        selection = self.robot_list.selection()
        if not selection:
            messagebox.showwarning("Warning", "Select a robot to edit.")
            return
        idx = int(selection[0])
        rob = self.robots[idx]
        for k in ['start_x','start_y','goal_x','goal_y','speed','start_time','deadline']:
            self.entries[k].delete(0, tk.END)
            val = rob.get(k, "")
            if val is None:
                val = ""
            self.entries[k].insert(0, str(val))
        self.edit_robot_index = idx
        self.save_edit_button.config(state=tk.NORMAL)
        self.cancel_edit_button.config(state=tk.NORMAL)

    def save_edit_robot(self):
        if self.edit_robot_index is None:
            return
        try:
            config = {
                'start_x': int(self.entries['start_x'].get()),
                'start_y': int(self.entries['start_y'].get()),
                'goal_x': int(self.entries['goal_x'].get()),
                'goal_y': int(self.entries['goal_y'].get()),
                'speed': int(self.entries['speed'].get()),
                'start_time': int(self.entries['start_time'].get()),
                'deadline': int(self.entries['deadline'].get()) if self.entries['deadline'].get().strip() else None
            }
            w, h = self.grid_size
            # Validate
            if not (0 <= config['start_x'] < h and 0 <= config['start_y'] < w):
                raise ValueError("Start out of bounds")
            if not (0 <= config['goal_x'] < h and 0 <= config['goal_y'] < w):
                raise ValueError("Goal out of bounds")

            self.robots[self.edit_robot_index] = config
            self.edit_robot_index = None
            self.update_robot_list()
            self.update_grid_display()
            self.clear_entries()
            self.save_edit_button.config(state=tk.DISABLED)
            self.cancel_edit_button.config(state=tk.DISABLED)
            messagebox.showinfo("Success", "Robot updated.")
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def cancel_edit_robot(self):
        self.edit_robot_index = None
        self.clear_entries()
        self.save_edit_button.config(state=tk.DISABLED)
        self.cancel_edit_button.config(state=tk.DISABLED)

    def remove_robot(self):
        selection = self.robot_list.selection()
        if not selection:
            messagebox.showwarning("Warning", "Select a robot to remove.")
            return
        idx = int(selection[0])
        del self.robots[idx]
        self.update_robot_list()
        self.update_grid_display()
        messagebox.showinfo("Removed", "Robot removed.")

    def update_robot_list(self):
        self.robot_list.delete(*self.robot_list.get_children())
        for i, r in enumerate(self.robots):
            vals = (
                f"({r['start_x']},{r['start_y']})",
                f"({r['goal_x']},{r['goal_y']})",
                r['speed'],
                r['start_time'],
                r['deadline'] if r['deadline'] is not None else '∞'
            )
            self.robot_list.insert('', 'end', iid=str(i), text=str(i+1), values=vals)

    def clear_entries(self):
        for e in self.entries.values():
            e.delete(0, tk.END)

    # ---------------------- DEFAULT SETUP ----------------------
    def load_default_setup(self):
        self.grid_size = (20, 20)  # width=20, height=20
        self.width_entry.set(self.grid_size[0])
        self.height_entry.set(self.grid_size[1])
        
        # Some obstacles (row, col)
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
        
        # Default robots
        self.robots = [
            {'start_x': 0,  'start_y':10, 'goal_x':18, 'goal_y':10, 'speed':1, 'start_time':0,  'deadline':50},
            {'start_x':18, 'start_y':9,  'goal_x':1,  'goal_y':11, 'speed':1, 'start_time':5,  'deadline':60},
            {'start_x':18, 'start_y':0,  'goal_x':0,  'goal_y':18, 'speed':1, 'start_time':0,  'deadline':40},
            {'start_x':0,  'start_y':18, 'goal_x':18, 'goal_y':0,  'speed':1, 'start_time':10, 'deadline':55},
            {'start_x':10, 'start_y':13, 'goal_x':18, 'goal_y':18, 'speed':1, 'start_time':0,  'deadline':35},
            {'start_x':6,  'start_y':12, 'goal_x':6,  'goal_y':12, 'speed':1, 'start_time':0,  'deadline':None},
        ]
        
        self.update_robot_list()
        self.update_grid_display()
        messagebox.showinfo("Default Setup Loaded", "Default obstacles and robots loaded.")
    
    # ---------------------- RUN SIMULATION ----------------------
    def run_simulation(self):
        # Prepare output directories
        output_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(output_dir, exist_ok=True)
        temp_dir = os.path.join(output_dir, "temp")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        w, h = self.grid_size
        # Build the occupancy grid (shape = (h, w))
        grid = np.zeros((h, w), dtype=int)
        for (r, c) in self.obstacles:
            if 0 <= r < h and 0 <= c < w:
                grid[r][c] = 1
        
        # Create Robot objects
        robots = []
        for i, cfg in enumerate(self.robots):
            rob = Robot(
                start_x=cfg['start_x'],
                start_y=cfg['start_y'],
                goal_x=cfg['goal_x'],
                goal_y=cfg['goal_y'],
                speed=cfg['speed'],
                start_time=cfg['start_time'],
                deadline=cfg['deadline']
            )
            rob.number = i+1
            robots.append(rob)
        
        images = []
        max_steps = 200
        log_path = os.path.join(output_dir, "simulation_log.txt")
        with open(log_path, 'w', encoding='utf-8') as log_file:
            for step in range(max_steps):
                log_file.write(f"\nStep {step}:\n")
                # Sort active robots by priority
                active_robots = [r for r in robots if step >= r.start_time and not r.at_goal()]
                active_robots.sort(key=Robot.get_priority)
                
                # Each active robot attempts to move
                for r in active_robots:
                    r.step(grid, robots, step)
                    log_file.write(f"Robot {r.number} at position ({r.x}, {r.y})\n")
                
                # Plot
                plt.figure(figsize=(8,8))
                plt.imshow(grid, cmap='binary', origin='upper', interpolation='nearest')
                colors = ["blue","green","red","purple","orange","brown","cyan","magenta"]
                
                for r in robots:
                    plt.plot(r.y, r.x, marker='o', color=colors[(r.number-1)%len(colors)], markersize=8)
                    label_text = f"S:{r.start_time},D:{r.deadline if r.deadline else '∞'}"
                    plt.text(r.y, r.x, label_text, color="black", fontsize=8, ha="center", va="bottom")
                
                plt.title(f"Step {step}")
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
                
                frame_path = os.path.join(temp_dir, f"step_{step:03d}.png")
                plt.savefig(frame_path)
                plt.close()
                
                images.append(Image.open(frame_path))
                
                # Check if all done or deadlines missed
                if all(r.at_goal() or r.deadline_missed for r in robots):
                    break
        
        # Build GIF
        if images:
            gif_path = os.path.join(output_dir, "robot_simulation.gif")
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=600,
                loop=0
            )
            self.show_gif(gif_path)
            self.status_label.config(text=f"Simulation complete. GIF at: {gif_path}")
            messagebox.showinfo("Done", f"Simulation complete.\nResults in {gif_path}")
            shutil.rmtree(temp_dir)
        else:
            messagebox.showwarning("No Images", "No simulation frames generated.")

    # ---------------------- GIF ANIMATION ----------------------
    def show_gif(self, path):
        self.gif_frames = []
        try:
            im = Image.open(path)
            for frame in ImageSequence.Iterator(im):
                frame = frame.resize((600,600), Image.Resampling.LANCZOS)
                self.gif_frames.append(ImageTk.PhotoImage(frame))
        except Exception as e:
            messagebox.showerror("GIF Error", str(e))
            return
        if self.gif_frames:
            self.animate(0)
    
    def animate(self, idx):
        self.gif_label.config(image=self.gif_frames[idx])
        idx = (idx+1) % len(self.gif_frames)
        self.root.after(600, self.animate, idx)

if __name__ == "__main__":
    root = tk.Tk()
    gui = SimulationGUI(root)
    root.geometry("1200x800")
    root.mainloop()
