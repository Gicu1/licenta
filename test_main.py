import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # Only needed if saving plots as specific image types other than default
import os
import time
import random
from heapq import heappush, heappop
from typing import List, Set, Tuple, Optional, Dict, Any

# Import the updated Robot class and type aliases
try:
    from Robot import Robot, Grid, State, Path
except ImportError:
    print("ERROR: Make sure 'Robot.py' is in the same directory or Python path.")
    sys.exit() # Use sys.exit() for cleaner exit in scripts

# -------------------- Constants --------------------
GRID_WIDTH = 50
GRID_HEIGHT = 50
NUM_ROBOTS_PER_TEST = 15 
TESTS_PER_TYPE = 20      # Number of random instances per scenario type
MAX_SIMULATION_STEPS = 300 # Max steps before simulation terminates
DEFAULT_OBSTACLE_RATIO = 0.25
HIGH_OBSTACLE_RATIO = 0.40
PLOTS_SUBDIR = "test_plots_worst" # Subdirectory to save result plots

# -------------------- Helper: Shortest Path (for Benchmarking) --------------------

def heuristic_grid(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Manhattan distance heuristic for grid coordinates (row, col)."""
    (r1, c1) = a
    (r2, c2) = b
    return abs(r1 - r2) + abs(c1 - c2)

def shortest_path_length(grid: Grid, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[int]:
    """
    Computes the shortest path length (number of steps) from start to goal
    on the grid using A*, ignoring other robots (static grid only).

    Args:
        grid (Grid): The static environment grid (0=free, 1=obstacle).
        start (Tuple[int, int]): Start position (row, col).
        goal (Tuple[int, int]): Goal position (row, col).

    Returns:
        Optional[int]: The length of the shortest path (number of steps),
                       or None if no path exists. Returns 0 if start == goal.
    """
    if start == goal:
        return 0

    height, width = grid.shape
    if not (0 <= start[0] < height and 0 <= start[1] < width and grid[start[0], start[1]] == 0):
        return None # Start is invalid
    if not (0 <= goal[0] < height and 0 <= goal[1] < width and grid[goal[0], goal[1]] == 0):
        return None # Goal is invalid

    # A* Search (simplified for path length only)
    # Priority queue: (f_score, g_score, position)
    open_set: List[Tuple[int, int, Tuple[int, int]]] = []
    heappush(open_set, (heuristic_grid(start, goal), 0, start)) # f = h, g = 0

    # g_score: cost (steps) from start to a position
    g_score_map: Dict[Tuple[int, int], int] = {start: 0}

    # Directions (dr, dc) for 4-directional movement
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] # Right, Down, Left, Up

    while open_set:
        f_val, g, current_pos = heappop(open_set)

        if current_pos == goal:
            return g # Found the shortest path length

        # Check if we found a shorter path already (acts like closed set)
        if g > g_score_map.get(current_pos, float('inf')):
            continue

        # Explore neighbors
        r, c = current_pos
        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            # Check bounds and obstacles
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] == 0:
                neighbor_pos = (nr, nc)
                tentative_g = g + 1

                # If this path to neighbor is shorter than any previous found path
                if tentative_g < g_score_map.get(neighbor_pos, float('inf')):
                    g_score_map[neighbor_pos] = tentative_g
                    f_new = tentative_g + heuristic_grid(neighbor_pos, goal)
                    heappush(open_set, (f_new, tentative_g, neighbor_pos))

    return None # Goal is unreachable

# -------------------- Plotting Function --------------------

def plot_map_with_paths(grid: Grid,
                        final_robots: List[Robot],
                        title: str = "Scenario Result",
                        save_path: Optional[str] = None):
    """
    Plots the grid map, obstacles, and the final state of each robot,
    including start, goal, final position, and planned path.

    Args:
        grid (Grid): The static environment grid.
        final_robots (List[Robot]): List of Robot objects in their final state after simulation.
        title (str): Title for the plot.
        save_path (Optional[str]): If provided, saves the plot to this file path.
    """
    plt.figure(figsize=(10, 10))
    height, width = grid.shape
    # Use binary for white background (free=0), black obstacles (obstacle=1)
    # origin='lower' matches numpy array index (row, col) to plot coords (y, x)
    plt.imshow(grid, cmap='binary', origin='lower', interpolation='nearest')

    colors = plt.cm.get_cmap('tab10', max(10, len(final_robots))) # Use a colormap

    # For legend clarity, label only the first instance of each marker type
    labelled_start, labelled_goal, labelled_end_ok, labelled_end_fail = False, False, False, False

    for i, robot in enumerate(final_robots):
        robot_id = robot.robot_id if robot.robot_id is not None else f"R{i+1}" # Handle potential None ID
        color = colors(i % 10) # Cycle through colors

        # --- Plot Path ---
        # Plot the path stored in the robot object. Caveat: This is the *last planned* path.
        # It might differ from actual execution if dynamic replanning occurred often.
        if robot.path and len(robot.path) > 1:
            # Path is list of (row, col, t). We need (col, row) for plotting with origin='lower'.
            path_cols = [state[1] for state in robot.path]
            path_rows = [state[0] for state in robot.path]
            plt.plot(path_cols, path_rows, marker='.', linestyle='-', color=color, alpha=0.5, markersize=3,
                     label=f'_path_{robot_id}') # Use _ prefix to hide path from main legend

        # --- Plot Markers ---
        start_pos = (robot.start_x, robot.start_y) # (row, col)
        goal_pos = (robot.goal_x, robot.goal_y)     # (row, col)
        final_pos = (robot.x, robot.y)             # (row, col)

        # Plot Start Position (Circle)
        start_label = 'Start' if not labelled_start else None
        plt.plot(start_pos[1], start_pos[0], marker='o', color=color, markersize=8, linestyle='None',
                 markeredgecolor='black', label=start_label)
        labelled_start = True
        plt.text(start_pos[1], start_pos[0], f"{robot_id}", color='white', fontsize=7,
                 ha='center', va='center', fontweight='bold')

        # Plot Goal Position (Star)
        goal_label = 'Goal' if not labelled_goal else None
        plt.plot(goal_pos[1], goal_pos[0], marker='*', color=color, markersize=12, linestyle='None',
                 markeredgecolor='black', label=goal_label)
        labelled_goal = True
        plt.text(goal_pos[1]+0.1, goal_pos[0]+0.1, f"{robot_id}", color='black', fontsize=7,
                 ha='left', va='bottom')


        # Plot Final Position (Square: Green=OK, Red=Failed)
        if robot.at_goal():
            end_marker = 's'
            end_color = 'lime'
            end_label = 'Final (Goal)' if not labelled_end_ok else None
            labelled_end_ok = True
            status = "Success"
        else:
            end_marker = 'X'
            end_color = 'red'
            end_label = 'Final (Failed)' if not labelled_end_fail else None
            labelled_end_fail = True
            status = "Failed"
            if robot.deadline_missed and robot.deadline is not None:
                status += f" (D:{robot.deadline})" # Add deadline info if missed
            elif robot.deadline_missed:
                 status += " (No Path/Stuck)" # Indicate other failure if deadline wasn't the cause
            elif robot.deadline is not None and robot.path and robot.path[-1][2] > robot.deadline:
                 status += f" (Path>D:{robot.deadline})" # Path itself exceeded deadline

        plt.plot(final_pos[1], final_pos[0], marker=end_marker, color=end_color, markersize=10, linestyle='None',
                 markeredgecolor='black', label=end_label)
        plt.text(final_pos[1]-0.1, final_pos[0]-0.1, f"{robot_id}", color='black', fontsize=7,
                 ha='right', va='top')

    plt.title(title, fontsize=10) # Slightly smaller title font if needed
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    # Adjust ticks for clarity on larger grids if needed
    if width > 30:
        plt.xticks(np.arange(0, width, 5))
    else:
        plt.xticks(np.arange(width))
    if height > 30:
        plt.yticks(np.arange(0, height, 5))
    else:
        plt.yticks(np.arange(height))

    plt.grid(True, which='both', color='lightgray', linestyle=':', linewidth=0.5)
    # Place legend outside plot area
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout further if legend overlaps

    if save_path:
        try:
            plt.savefig(save_path, bbox_inches='tight', dpi=150) # Increase DPI for better quality if needed
            print(f"Plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}", file=sys.stderr)
    else:
        plt.show()
    plt.close() # Close the figure to free memory

# -------------------- MAP GENERATORS --------------------

def generate_random_map(width: int, height: int, obstacle_ratio: float = DEFAULT_OBSTACLE_RATIO) -> Grid:
    """Generate a random grid with a given obstacle ratio."""
    grid = np.zeros((height, width), dtype=int)
    num_cells = width * height
    num_obstacles = int(obstacle_ratio * num_cells)
    if num_obstacles >= num_cells: return np.ones((height, width), dtype=int) # Avoid sampling errors

    # Generate obstacle coordinates
    obstacle_indices = random.sample(range(num_cells), num_obstacles)
    obstacle_coords = [(idx // width, idx % width) for idx in obstacle_indices]

    for r, c in obstacle_coords:
        grid[r, c] = 1
    return grid

def generate_dense_map(width: int, height: int) -> Grid:
    """Generate a grid with high obstacle density."""
    return generate_random_map(width, height, obstacle_ratio=HIGH_OBSTACLE_RATIO)

def generate_corridor_map(width: int, height: int) -> Grid:
    """Generate a map with a vertical and horizontal corridor."""
    grid = np.ones((height, width), dtype=int)
    # Vertical corridor
    corridor_width_v = max(1, width // 8)
    start_col_v = width // 2 - corridor_width_v // 2
    grid[:, start_col_v : start_col_v + corridor_width_v] = 0
    # Horizontal corridor
    corridor_width_h = max(1, height // 8)
    start_row_h = height // 2 - corridor_width_h // 2
    grid[start_row_h : start_row_h + corridor_width_h, :] = 0
    return grid

def generate_bottleneck_map(width: int, height: int) -> Grid:
    """Generate a map with two open areas connected by a narrow bottleneck."""
    grid = np.ones((height, width), dtype=int)
    center_col = width // 2
    bottleneck_width = 1 # Only one cell wide
    # Open areas on left and right
    grid[:, :center_col - 1] = 0 # Leave wall column center_col - 1
    grid[:, center_col + bottleneck_width :] = 0 # Leave wall column center_col
    # Create the bottleneck passage(s) - e.g., one in the middle row
    passage_row = height // 2
    grid[passage_row, center_col -1 : center_col + bottleneck_width] = 0
    # Optionally add another passage
    # grid[passage_row + 2, center_col -1 : center_col + bottleneck_width] = 0
    return grid

# -------------------- ROBOT GENERATION --------------------

def generate_robots_for_map(grid: Grid,
                            num_robots: int,
                            scenario_type: str,
                            deadline_tightness_prob: float = 0.1) -> Optional[List[Robot]]:
    """
    Generates a list of Robot objects for a given grid and scenario type.
    Ensures start/goal pairs are valid, reachable, and distinct.

    Args:
        grid (Grid): The environment grid.
        num_robots (int): The number of robots to generate.
        scenario_type (str): Type of scenario (e.g., "random", "deadline_stress").
                             Used for potential scenario-specific logic.
        deadline_tightness_prob (float): For 'deadline_stress', probability of assigning a very tight deadline.

    Returns:
        Optional[List[Robot]]: A list of initialized Robot objects, or None if generation fails.
    """
    height, width = grid.shape
    free_cells = [(r, c) for r in range(height) for c in range(width) if grid[r, c] == 0]

    if len(free_cells) < num_robots * 2:
        print(f"Error: Not enough free cells ({len(free_cells)}) to place {num_robots} robots (need {num_robots*2}).", file=sys.stderr)
        return None

    robots: List[Robot] = []
    used_starts_goals: Set[Tuple[int, int]] = set() # Explicitly type hint
    attempts = 0
    max_attempts = num_robots * 100 # Increase attempts for potentially harder scenarios

    while len(robots) < num_robots and attempts < max_attempts:
        attempts += 1
        try:
            start, goal = random.sample(free_cells, 2)
        except ValueError:
             print("Error: Could not sample 2 distinct free cells.", file=sys.stderr)
             return None # Should not happen based on earlier check, but safeguard

        # Ensure start/goal not already used by another robot in this set
        if start in used_starts_goals or goal in used_starts_goals:
            continue

        # Check reachability and get shortest path length for deadline calculation
        sp_len = shortest_path_length(grid, start, goal)
        if sp_len is None:
            continue # Unreachable pair, try again

        # Determine deadline based on scenario type
        deadline: Optional[int] = None
        start_time = 0 # Default start time to 0 for tests

        if scenario_type == "deadline_stress":
            # Assign tighter deadlines more often in stress tests
            if random.random() < deadline_tightness_prob:
                 # Very tight deadline: SP + 0 to 2 extra steps
                 deadline = sp_len + random.randint(0, max(0, min(2, sp_len // 5))) # Ensure non-negative buffer
            else:
                 # Moderate deadline: SP + small buffer
                 deadline = sp_len + random.randint(max(1, sp_len // 4), max(5, sp_len // 2))
        elif scenario_type == "random":
             # Give a reasonable buffer, or no deadline sometimes
             if random.random() < 0.7: # 70% chance of having a deadline
                 # Generous deadline: SP + larger buffer
                 deadline = sp_len + random.randint(max(3, sp_len // 2), max(15, int(sp_len * 1.5)))
             else:
                  deadline = None # No deadline
        else: # Default for corridor, bottleneck etc. - moderate deadline
            deadline = sp_len + random.randint(max(2, sp_len // 3), max(10, sp_len))

        # Ensure deadline > start_time (if start_time wasn't 0)
        if deadline is not None and deadline <= start_time:
             deadline = start_time + sp_len + 1 # Basic feasible deadline

        # Create robot instance
        robot_id = len(robots) + 1 # Simple 1-based ID for testing
        r = Robot(
            robot_id=robot_id,
            start_x=start[0], start_y=start[1],
            goal_x=goal[0], goal_y=goal[1],
            start_time=start_time,
            deadline=deadline
        )
        robots.append(r)
        used_starts_goals.add(start)
        used_starts_goals.add(goal)

    if len(robots) < num_robots:
        print(f"Warning: Failed to generate {num_robots} valid robot configurations after {max_attempts} attempts. Generated {len(robots)}.", file=sys.stderr)
        if not robots:
             return None # Return None if no robots could be generated
        # Proceed with the robots that were generated

    return robots

# -------------------- Simulation Function --------------------

def simulate_scenario(grid: Grid,
                      initial_robots: List[Robot],
                      max_steps: int = MAX_SIMULATION_STEPS,
                      logging: bool = False) -> Tuple[List[Robot], int]:
    """
    Runs the simulation for a given scenario.

    Args:
        grid (Grid): The environment grid.
        initial_robots (List[Robot]): List of pre-configured Robot objects.
        max_steps (int): Maximum simulation steps.
        logging (bool): If True, prints step-by-step log to console.

    Returns:
        Tuple[List[Robot], int]:
            - List of Robot objects in their final state.
            - Number of steps the simulation actually ran.
    """
    # CRITICAL: Clear shared state BEFORE this specific simulation run
    Robot.clear_reservations()

    # Work on a copy of the robot list if needed, though Robot class manages its own state
    robots = initial_robots # Use the provided list directly
    num_robots = len(robots)
    if not robots:
        return [], 0 # Handle case with no robots generated

    if logging: print(f"\n--- Starting Simulation: {num_robots} robots ---")

    actual_steps = 0
    for step in range(max_steps):
        actual_steps = step
        if logging: print(f"\n--- Step {step} ---")

        # Determine active robots (not finished, not failed, started)
        # Important: Ensure robot has started before considering it "inactive" due to failure/goal
        active_robots_this_step = [
            r for r in robots if step >= r.start_time and not r.at_goal() and not r.deadline_missed
        ]

        # Check termination condition BEFORE processing step
        # Simulation ends if NO robots are currently active AND all robots that *should* have started
        # are either at their goal or have failed.
        should_be_active_or_done = [r for r in robots if step >= r.start_time]
        if not active_robots_this_step and all(r.at_goal() or r.deadline_missed for r in should_be_active_or_done):
            if logging: print(f"Simulation End: All started robots finished or failed at step {step}.")
            break # End simulation early

        # Sort active robots by priority (lower value = higher priority)
        active_robots_this_step.sort(key=lambda r: r.get_priority(step))

        if logging:
            order = [r.robot_id for r in active_robots_this_step]
            print(f"Processing order: {order}")

        # Let each active robot attempt its step
        for robot in active_robots_this_step:
            try:
                robot.step(grid, robots, step) # Pass full robot list if needed by step logic
                if logging: print(f"  {robot}") # Print state after step attempt
            except Exception as e:
                 # Log error and mark robot as failed to prevent simulation crash
                 print(f"\nCRITICAL ERROR during step() for Robot {robot.robot_id} at step {step}: {e}", file=sys.stderr)
                 import traceback
                 traceback.print_exc(file=sys.stderr)
                 robot.deadline_missed = True # Mark robot as failed if step crashes
                 robot.path = [] # Clear path as it's likely invalid
                 robot.remove_reservations() # Clean up reservations

    else: # Loop finished without break (max_steps reached)
        if logging: print(f"\nSimulation End: Max steps ({max_steps}) reached.")
        actual_steps = max_steps # Indicate max steps were run

    # Return the list of robots in their final state and steps run (use actual_steps + 1 if 0-indexed)
    # If loop broke, actual_steps is the step it broke on. If loop finished, it's max_steps-1.
    # We want the total *number* of steps simulated.
    final_step_count = actual_steps if actual_steps == max_steps else actual_steps + 1
    return robots, final_step_count

# -------------------- Test Harness --------------------

def run_in_depth_tests():
    """
    Runs a suite of tests across different map types and robot configurations,
    collects metrics, and plots the **worst** result for each scenario type.
    """
    scenario_map_generators: Dict[str, callable] = {
        "random": lambda w, h: generate_random_map(w, h, DEFAULT_OBSTACLE_RATIO),
        "dense": generate_dense_map,
        "corridor": generate_corridor_map,
        "bottleneck": generate_bottleneck_map,
        "deadline_stress": lambda w, h: generate_random_map(w, h, DEFAULT_OBSTACLE_RATIO), # Use random map for stress
    }
    num_scenario_types = len(scenario_map_generators)
    total_tests_to_run = num_scenario_types * TESTS_PER_TYPE

    # Aggregate statistics
    overall_stats = {
        "robots_tested": 0,
        "successes": 0,
        "total_path_steps": 0,
        "total_sp_steps": 0,
        "valid_ratios": 0, # Count of successful robots with valid SP > 0
        "total_sim_time": 0.0,
        "total_sim_steps": 0,
        "tests_completed": 0,
        "tests_failed_generation": 0,
    }
    # Store results per type for finding the worst run
    results_by_type: Dict[str, List[Dict[str, Any]]] = {stype: [] for stype in scenario_map_generators}
    # --- Store data for the WORST plot per type ---
    worst_scenario_plots: Dict[str, Dict[str, Any]] = {}

    print("\n===== MAPF Algorithm Testing Suite (Finding WORST Runs) =====")
    print(f"Grid Size: {GRID_HEIGHT}x{GRID_WIDTH}")
    print(f"Robots per Test: {NUM_ROBOTS_PER_TEST}")
    print(f"Tests per Scenario Type: {TESTS_PER_TYPE}")
    print(f"Max Simulation Steps: {MAX_SIMULATION_STEPS}")
    print("===========================================================\n")

    # Prepare plot directory
    if not os.path.exists(PLOTS_SUBDIR):
        os.makedirs(PLOTS_SUBDIR)
        print(f"Created directory: {PLOTS_SUBDIR}")
    else:
        print(f"Using existing directory: {PLOTS_SUBDIR}")


    test_count = 0
    for s_type, map_generator in scenario_map_generators.items():
        print(f"\n--- Testing Scenario Type: {s_type.upper()} ---")
        type_successes = 0
        type_robots = 0
        worst_run_for_this_type: Optional[Dict[str, Any]] = None # Track worst run within this type

        for test_num in range(TESTS_PER_TYPE):
            test_count += 1
            print(f"  Running Test {test_num + 1}/{TESTS_PER_TYPE} ({test_count}/{total_tests_to_run})... ", end="")
            start_time = time.time()

            # 1. Generate Map
            grid = map_generator(GRID_WIDTH, GRID_HEIGHT)

            # 2. Generate Robots
            initial_robots = generate_robots_for_map(grid, NUM_ROBOTS_PER_TEST, s_type)
            if initial_robots is None:
                print("FAILED (Robot Generation Error)")
                overall_stats["tests_failed_generation"] += 1
                continue # Skip this test instance

            # 3. Run Simulation
            final_robots, steps_run = simulate_scenario(grid, initial_robots, MAX_SIMULATION_STEPS, logging=False)
            sim_time = time.time() - start_time
            overall_stats["tests_completed"] += 1

            # 4. Calculate Metrics for this test run
            num_robots = len(final_robots)
            successes = sum(1 for r in final_robots if r.at_goal())
            type_successes += successes
            type_robots += num_robots

            current_test_ratio_sum = 0.0
            current_test_valid_ratios = 0
            current_test_sp_sum = 0 # Sum of SP lengths for valid paths
            current_test_actual_sum = 0 # Sum of actual path lengths for valid paths

            for robot in final_robots:
                if robot.at_goal() and robot.path and len(robot.path) > 1:
                    # Path length is number of states - 1 = number of steps taken
                    actual_steps = robot.total_moves
                    sp = shortest_path_length(grid, (robot.start_x, robot.start_y), (robot.goal_x, robot.goal_y))
                    if sp is not None and sp > 0:
                        ratio = actual_steps / sp
                        current_test_ratio_sum += ratio
                        current_test_valid_ratios += 1
                        current_test_sp_sum += sp
                        current_test_actual_sum += actual_steps
                        # Add to overall stats only if valid
                        overall_stats["total_path_steps"] += actual_steps
                        overall_stats["total_sp_steps"] += sp
                        overall_stats["valid_ratios"] += 1
                    elif sp == 0: # Handle start==goal case (should have ratio 1 if successful)
                         current_test_ratio_sum += 1.0 # Assume ratio 1 if SP is 0 and robot succeeded
                         current_test_valid_ratios += 1
                         overall_stats["valid_ratios"] += 1
                         # No steps added to overall path/sp steps

            # Calculate average ratio specifically for this test run
            avg_ratio_this_test = current_test_ratio_sum / current_test_valid_ratios if current_test_valid_ratios > 0 else None
            success_rate_this_test = successes / num_robots if num_robots > 0 else 0.0 # Ensure float division or 0.0

            print(f"Done ({sim_time:.2f}s). Success: {successes}/{num_robots} ({success_rate_this_test*100:.0f}%). Avg Ratio: {f'{avg_ratio_this_test:.2f}' if avg_ratio_this_test else 'N/A'}. Steps: {steps_run}.")

            # Store results for this test run
            test_result_data = {
                "success_rate": success_rate_this_test,
                "avg_ratio": avg_ratio_this_test, # Average ratio for *this test run*
                "success_count": successes,
                "robot_count": num_robots,
                "sim_time": sim_time,
                "steps_run": steps_run,
                "grid": grid,
                "final_robots": final_robots,
                "test_num": test_num + 1
            }
            results_by_type[s_type].append(test_result_data)

            # Update overall stats
            overall_stats["robots_tested"] += num_robots
            overall_stats["successes"] += successes
            overall_stats["total_sim_time"] += sim_time
            overall_stats["total_sim_steps"] += steps_run

            # --- Track WORST scenario for plotting ---
            # Definition of worst: Lowest success rate, then highest average ratio as tie-breaker.
            if worst_run_for_this_type is None:
                worst_run_for_this_type = test_result_data # First run is the worst so far
            else:
                new_success_rate = test_result_data["success_rate"]
                worst_success_rate = worst_run_for_this_type["success_rate"]

                # Treat None ratio as -infinity when seeking highest ratio (less bad than any positive ratio)
                new_ratio = test_result_data["avg_ratio"] if test_result_data["avg_ratio"] is not None else -float('inf')
                worst_ratio = worst_run_for_this_type["avg_ratio"] if worst_run_for_this_type["avg_ratio"] is not None else -float('inf')

                is_objectively_worse = False
                if new_success_rate < worst_success_rate:
                    is_objectively_worse = True
                elif new_success_rate == worst_success_rate:
                    # If success rates are equal, the one with the HIGHER ratio is worse
                    if new_ratio > worst_ratio:
                        is_objectively_worse = True


                if is_objectively_worse:
                    worst_run_for_this_type = test_result_data

        # Store the identified worst run for this scenario type
        if worst_run_for_this_type:
            worst_scenario_plots[s_type] = worst_run_for_this_type

        # Print summary for the scenario type
        type_success_rate = type_successes / type_robots if type_robots > 0 else 0
        print(f"  -- {s_type.upper()} Summary: Avg Success Rate: {type_success_rate * 100:.1f}% ({type_successes}/{type_robots}) --")


    # --- Overall Summary ---
    print("\n===== Overall Test Summary =====")
    actual_tests_run = overall_stats["tests_completed"]
    overall_success_rate = overall_stats["successes"] / overall_stats["robots_tested"] if overall_stats["robots_tested"] > 0 else 0
    avg_sim_time = overall_stats["total_sim_time"] / actual_tests_run if actual_tests_run > 0 else 0
    avg_sim_steps = overall_stats["total_sim_steps"] / actual_tests_run if actual_tests_run > 0 else 0

    # Calculate overall average optimality ratio carefully using aggregated sums
    overall_avg_ratio = (overall_stats["total_path_steps"] / overall_stats["total_sp_steps"]
                         if overall_stats["valid_ratios"] > 0 and overall_stats["total_sp_steps"] > 0 else None)

    print(f"Total Tests Attempted: {total_tests_to_run}")
    print(f"Total Tests Completed: {actual_tests_run}")
    if overall_stats["tests_failed_generation"] > 0:
         print(f"Tests Failed (Robot Gen): {overall_stats['tests_failed_generation']}")
    print(f"Total Robots Tested: {overall_stats['robots_tested']}")
    print(f"Overall Success Rate: {overall_success_rate * 100:.2f}% ({overall_stats['successes']} successful robots)")
    if overall_avg_ratio is not None:
        print(f"Overall Average Optimality Ratio (Successful Robots): {overall_avg_ratio:.3f}")
        print(f"  (Based on {overall_stats['valid_ratios']} successful robot paths with SP > 0)")
    else:
         print("Overall Average Optimality Ratio: N/A (No successful paths with SP>0 recorded)")
    print(f"Average Simulation Time per Completed Test: {avg_sim_time:.3f} seconds")
    print(f"Average Simulation Steps per Completed Test: {avg_sim_steps:.1f} steps")
    print("===============================\n")


    # --- Plot WORST Scenarios ---
    print("Generating plots for the WORST run of each scenario type...")
    if not worst_scenario_plots:
        print("No worst scenario plots to generate (perhaps no tests completed successfully).")
    else:
        for s_type, worst_data in worst_scenario_plots.items():
            avg_ratio_str = f"{worst_data['avg_ratio']:.2f}" if worst_data['avg_ratio'] is not None else 'N/A'
            plot_title = (f"Worst Run: {s_type.capitalize()} Scenario (Test {worst_data['test_num']})\n"
                          f"Success: {worst_data['success_count']}/{worst_data['robot_count']} ({worst_data['success_rate']*100:.0f}%), "
                          f"Avg Ratio: {avg_ratio_str}, "
                          f"Time: {worst_data['sim_time']:.2f}s, Steps: {worst_data['steps_run']}")
            plot_filename = os.path.join(PLOTS_SUBDIR, f"worst_{s_type}_scenario.png")
            plot_map_with_paths(worst_data["grid"],
                                worst_data["final_robots"],
                                title=plot_title,
                                save_path=plot_filename)
        print(f"Worst run plots saved in: {PLOTS_SUBDIR}")

    print("\nTesting finished.")

# -------------------- Main Execution Guard --------------------
if __name__ == "__main__":
    run_in_depth_tests()