import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
import time
import random
from heapq import heappush, heappop
from Robot import Robot  # Your improved Robot class

# -------------------- HELPER FUNCTIONS --------------------

def heuristic(a, b):
    # Chebyshev distance is admissible for 8-directional movement.
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def shortest_path_length(grid, start, goal):
    """Compute the shortest path length from start to goal on the grid using A*.
       Returns the number of steps or None if no path exists."""
    if start == goal:
        return 0
    height, width = grid.shape
    open_set = []
    heappush(open_set, (heuristic(start, goal), 0, start))
    g_score = {start: 0}
    closed_set = set()
    directions = [
        (0, 1), (1, 0), (0, -1), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]
    while open_set:
        f_val, g, (x, y) = heappop(open_set)
        if (x, y) == goal:
            return g
        closed_set.add((x, y))
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width and grid[nx][ny] == 0:
                if (nx, ny) in closed_set:
                    continue
                tentative_g = g + 1
                if tentative_g < g_score.get((nx, ny), float('inf')):
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + heuristic((nx, ny), goal)
                    heappush(open_set, (f, tentative_g, (nx, ny)))
    return None

def plot_map_with_paths(grid, robots, final_paths, title="Scenario"):
    """
    Plot the grid map and overlay each robot's path along with detailed annotations.
    For each robot, the following are shown:
      - Start position: Green circle.
      - Desired goal position: Red star.
      - Actual final state: Blue square if not reached, lime square if reached.
      - A text annotation showing deadline, start time, and status (Reached, Deadline Missed, or Stuck).
    """
    plt.figure(figsize=(10, 10))
    height, width = grid.shape
    plt.imshow(grid, cmap='gray_r')
    
    # For better legend control, we only label one instance per marker type.
    start_label, goal_label, end_label = True, True, True
    
    for i, robot in enumerate(robots):
        path = final_paths[i]
        x_coords = [pos[1] for pos in path]
        y_coords = [pos[0] for pos in path]
        plt.plot(x_coords, y_coords, marker='o', linestyle='-', label=f'Robot {robot.number} path')
        
        # Start position: first point of the robot's path.
        start_pos = (robot.path[0][0], robot.path[0][1])
        if start_label:
            plt.scatter(start_pos[1], start_pos[0], c='green', marker='o', s=100, edgecolors='black', label='Start')
            start_label = False
        else:
            plt.scatter(start_pos[1], start_pos[0], c='green', marker='o', s=100, edgecolors='black')
        
        # Desired goal position (from robot config)
        desired_goal = (robot.goal_x, robot.goal_y)
        if goal_label:
            plt.scatter(desired_goal[1], desired_goal[0], c='red', marker='*', s=150, edgecolors='black', label='Goal')
            goal_label = False
        else:
            plt.scatter(desired_goal[1], desired_goal[0], c='red', marker='*', s=150, edgecolors='black')
        
        # Actual end state (final state after simulation)
        end_pos = (robot.x, robot.y)
        color = 'lime' if robot.at_goal() else 'blue'
        if end_label:
            plt.scatter(end_pos[1], end_pos[0], c=color, marker='s', s=100, edgecolors='black', label='Final State')
            end_label = False
        else:
            plt.scatter(end_pos[1], end_pos[0], c=color, marker='s', s=100, edgecolors='black')
        
        # Determine status message
        if robot.at_goal():
            status = "Reached"
        elif robot.deadline_missed:
            status = "Deadline Missed"
        else:
            status = "Stuck"
        
        # Annotate near the final state with deadline and start info
        annotation = f"D:{robot.deadline} | S:{robot.start_time} | {status}"
        plt.text(end_pos[1], end_pos[0], annotation, fontsize=8, color='black',
                 ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# -------------------- MAP GENERATORS --------------------

def generate_random_map(width, height, obstacle_ratio=0.3):
    """Generate a random grid where 0=free and 1=obstacle."""
    grid = np.zeros((height, width), dtype=int)
    num_cells = width * height
    num_obstacles = int(obstacle_ratio * num_cells)
    obstacles = random.sample([(i, j) for i in range(height) for j in range(width)], num_obstacles)
    for (i, j) in obstacles:
        grid[i][j] = 1
    return grid

def generate_dense_map(width, height):
    """Generate a grid with a high obstacle density (e.g. 50% obstacles)."""
    return generate_random_map(width, height, obstacle_ratio=0.5)

def generate_narrow_corridor_map(width, height):
    """
    Create a grid mostly filled with obstacles except for a narrow vertical corridor
    and an additional horizontal passage to challenge path planning.
    """
    grid = np.ones((height, width), dtype=int)
    corridor_width = max(1, width // 10)
    start_col = width // 2 - corridor_width // 2
    grid[:, start_col:start_col+corridor_width] = 0
    grid[height // 2, :] = 0
    return grid

def generate_bottleneck_map(width, height):
    """
    Generate a map with two open areas connected by a narrow bottleneck.
    """
    grid = np.ones((height, width), dtype=int)
    grid[:, :width // 2 - 1] = 0
    grid[:, width // 2 + 1:] = 0
    for i in range(height):
        grid[i, width // 2] = 0
    return grid

def generate_deadline_stress_map(width, height, obstacle_ratio=0.3):
    """
    Generate a random map intended for deadline stress tests.
    """
    return generate_random_map(width, height, obstacle_ratio)

def generate_corner_case_map(width, height):
    """
    Generate a grid where obstacles force a long detour between corners.
    """
    grid = np.zeros((height, width), dtype=int)
    for i in range(height // 3, 2 * height // 3):
        grid[i, 1:width - 1] = 1
    grid[height // 2, width // 2] = 0
    return grid

# -------------------- ROBOT GENERATION --------------------

def generate_robots_for_map(grid, num_robots, scenario_type):
    """
    Generate robots on the provided grid. For each robot, choose two distinct free cells.
    For 'deadline_stress', sometimes assign a very tight deadline.
    """
    height, width = grid.shape
    free_cells = [(i, j) for i in range(height) for j in range(width) if grid[i][j] == 0]
    if len(free_cells) < 2 * num_robots:
        raise ValueError("Not enough free cells to place all robots.")
    robots = []
    trials = 0
    max_trials = 10000
    while len(robots) < num_robots and trials < max_trials:
        trials += 1
        start, goal = random.sample(free_cells, 2)
        sp = shortest_path_length(grid, start, goal)
        if sp is None:
            continue
        if scenario_type == "deadline_stress":
            if random.random() < 0.5:
                deadline = int(sp) + random.randint(0, 2)
            else:
                deadline = int(sp) + random.randint(5, 15)
        else:
            buffer_time = 10
            deadline = int(sp) + buffer_time
        r = Robot(start_x=start[0],
                  start_y=start[1],
                  goal_x=goal[0],
                  goal_y=goal[1],
                  start_time=0,
                  deadline=deadline)
        robots.append(r)
    if len(robots) < num_robots:
        raise RuntimeError("Could not generate enough valid robots for the scenario.")
    for i, robot in enumerate(robots):
        robot.number = i + 1
    return robots

# -------------------- SIMULATION FUNCTION --------------------

def simulate_scenario(grid, robots, max_steps=200, logging=False):
    """
    Run the simulation until all robots reach their goal, miss their deadlines,
    or max_steps is reached.
    Returns the final paths and a list indicating if each robot reached its goal.
    """
    grid_copy = np.copy(grid)
    num_robots = len(robots)
    log_lines = []
    for step in range(max_steps):
        if logging:
            log_lines.append(f"\nStep {step + 1}\n")
        moves_made = False
        active_robots = [r for r in robots if step >= r.start_time and not (r.at_goal() or r.deadline_missed)]
        active_robots.sort(key=Robot.get_priority)
        for robot in active_robots:
            moved = robot.step(grid_copy, robots, step)
            moves_made = moves_made or moved
            if logging:
                log_lines.append(f"Robot {robot.number} position: ({robot.x}, {robot.y})\n")
        if all(r.at_goal() or r.deadline_missed for r in robots):
            if logging:
                log_lines.append("All robots reached their goals or missed their deadlines!\n")
            break
        if not moves_made:
            if logging:
                log_lines.append("No moves possible -- simulation appears to be stuck.\n")
            break
    if logging:
        print("".join(log_lines))
    final_paths = [r.path for r in robots]
    reached_goals = [r.at_goal() for r in robots]
    return final_paths, reached_goals

# -------------------- TEST HARNESS --------------------

def run_in_depth_tests():
    """
    Run a series of tests over several scenario types:
      - random, dense, narrow_corridor, bottleneck, deadline_stress, and corner_case.
    For each scenario, run several tests and log metrics including:
      - Robots reaching goals,
      - Average optimality ratio (actual steps / shortest path),
      - Simulation time.
    The best scenario for each type (lowest average ratio) is plotted at the end with detailed annotations.
    """
    scenario_types = ["random", "dense", "narrow_corridor", "bottleneck", "deadline_stress", "corner_case"]
    tests_per_type = 5  # Number of tests per scenario type
    test_results = {}
    overall_success = 0
    overall_robots = 0
    overall_ratios = []
    overall_times = []
    best_scenarios = {}

    print("\n===== In-Depth MAPF Testing =====\n")
    for s_type in scenario_types:
        test_results[s_type] = []
        print(f"\n--- Testing Scenario Type: {s_type} ---")
        for test in range(tests_per_type):
            if s_type == "random":
                grid = generate_random_map(50, 50, obstacle_ratio=0.3)
            elif s_type == "dense":
                grid = generate_dense_map(50, 50)
            elif s_type == "narrow_corridor":
                grid = generate_narrow_corridor_map(50, 50)
            elif s_type == "bottleneck":
                grid = generate_bottleneck_map(50, 50)
            elif s_type == "deadline_stress":
                grid = generate_deadline_stress_map(50, 50, obstacle_ratio=0.3)
            elif s_type == "corner_case":
                grid = generate_corner_case_map(50, 50)
            else:
                grid = generate_random_map(50, 50, obstacle_ratio=0.3)
            
            try:
                robots = generate_robots_for_map(grid, num_robots=5, scenario_type=s_type)
            except Exception as e:
                print(f"Test {test+1} for {s_type}: Failed to generate robots: {e}")
                continue
            
            start = time.time()
            final_paths, reached_goals = simulate_scenario(grid, robots, max_steps=200, logging=False)
            sim_time = time.time() - start
            overall_times.append(sim_time)
            
            ratios = []
            for i, robot in enumerate(robots):
                actual_steps = len(final_paths[i]) - 1
                sp = shortest_path_length(grid, (final_paths[i][0][0], final_paths[i][0][1]), (robot.goal_x, robot.goal_y))
                if sp is not None and sp > 0:
                    ratios.append(actual_steps / sp)
            avg_ratio = np.mean(ratios) if ratios else None
            
            successes = sum(reached_goals)
            overall_success += successes
            overall_robots += len(robots)
            if ratios:
                overall_ratios.extend(ratios)
            
            test_results[s_type].append({
                "successes": successes,
                "ratios": ratios,
                "avg_ratio": avg_ratio,
                "sim_time": sim_time,
                "grid": grid,
                "robots": robots,
                "paths": final_paths,
                "reached_goals": reached_goals
            })
            
            print(f"Test {test+1} for {s_type}: {successes}/5 reached goal; "
                  f"Avg Optimality Ratio: {avg_ratio:.2f} ; Sim Time: {sim_time:.2f}s")
            
            if avg_ratio is not None:
                if s_type not in best_scenarios or best_scenarios[s_type]["avg_ratio"] > avg_ratio:
                    best_scenarios[s_type] = {
                        "avg_ratio": avg_ratio,
                        "grid": grid,
                        "robots": robots,
                        "paths": final_paths,
                        "reached_goals": reached_goals
                    }
    
    overall_success_rate = overall_success / overall_robots if overall_robots > 0 else 0
    overall_avg_ratio = np.mean(overall_ratios) if overall_ratios else None
    overall_avg_time = np.mean(overall_times) if overall_times else None
    
    print("\n===== Overall Summary =====")
    print(f"Total Robots Tested: {overall_robots}")
    print(f"Overall Success Rate: {overall_success_rate * 100:.2f}%")
    if overall_avg_ratio is not None:
        print(f"Overall Average Optimality Ratio: {overall_avg_ratio:.2f}")
    print(f"Overall Average Simulation Time: {overall_avg_time:.2f} seconds")
    
    # Plot best scenario for each scenario type with detailed annotations.
    for s_type, details in best_scenarios.items():
        title = f"Best Scenario for {s_type} (Avg Optimality Ratio: {details['avg_ratio']:.2f})"
        plot_map_with_paths(details["grid"], details["robots"], details["paths"], title=title)

if __name__ == "__main__":
    run_in_depth_tests()
