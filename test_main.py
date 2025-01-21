import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.ticker import MaxNLocator
import os
import shutil
import time
from Robot import Robot
from collections import deque
import random
from heapq import heappush, heappop

def plot_map_with_paths(grid, robots, final_paths):
    """
    Plot the grid map and overlay the paths of the robots.
    """
    plt.figure(figsize=(10,10))
    height, width = grid.shape
    # Plot the grid
    plt.imshow(grid, cmap='gray_r')
    # Plot the paths
    for i, path in enumerate(final_paths):
        x_coords = [pos[1] for pos in path]
        y_coords = [pos[0] for pos in path]
        plt.plot(x_coords, y_coords, label=f'Robot {i}')
        plt.scatter(x_coords[0], y_coords[0], c='green', marker='o')  # Start
        plt.scatter(x_coords[-1], y_coords[-1], c='red', marker='x')  # Goal
    plt.legend()
    plt.title('Map with Paths of Robots')
    plt.show()

def simulate_scenario(grid, robots, max_steps=100, show_path=False, logging=False):
    """
    Runs the MAPF simulation on a given grid and set of robots.
    Returns:
        final_paths: list of (x,y) positions for each robot
        reached_goals: list of booleans indicating if each robot reached its goal
    """
    # Copy grid to avoid modifying original
    grid = np.copy(grid)
    num_robots = len(robots)

    if logging:
        log_str = []
    
    for step in range(max_steps):
        if logging:
            log_str.append(f"\nStep {step+1}\n")

        moves_made = False
        for i, robot in enumerate(robots):
            moved = robot.step(grid, robots, step)
            moves_made = moves_made or moved
            if logging:
                log_str.append(f"Robot {i} position: ({robot.x}, {robot.y})\n")

        # Check if all robots reached their goals
        if all((robot.x, robot.y) == (robot.goal_x, robot.goal_y) for robot in robots):
            if logging:
                log_str.append("All robots reached their goals!\n")
            break

        # Check if no moves were made
        if not moves_made:
            if logging:
                log_str.append("No moves possible - simulation stuck\n")
            break

    if logging:
        print("".join(log_str))

    final_paths = [robot.path for robot in robots]
    reached_goals = [ (r.x, r.y) == (r.goal_x, r.goal_y) for r in robots ]

    return final_paths, reached_goals


def generate_random_map(width, height, obstacle_ratio=0.2):
    """Generate a random grid map with given obstacle ratio."""
    grid = np.zeros((height, width), dtype=int)
    num_obstacles = int(obstacle_ratio * width * height)
    obstacles = random.sample([(x, y) for x in range(height) for y in range(width)], num_obstacles)
    for (x, y) in obstacles:
        grid[x][y] = 1
    return grid

def generate_random_robots(grid, num_robots):
    """Generate random robots with start and goal positions on free cells."""
    height, width = grid.shape
    free_cells = [(x,y) for x in range(height) for y in range(width) if grid[x][y] == 0]

    # Each robot needs two distinct free cells: start and goal
    # We'll randomize and pick pairs
    if 2*num_robots > len(free_cells):
        raise ValueError("Not enough free cells to place robots.")
    random.shuffle(free_cells)

    robots = []
    for i in range(num_robots):
        start = free_cells.pop()
        goal = free_cells.pop()
        r = Robot(start_x=start[0], start_y=start[1], goal_x=goal[0], goal_y=goal[1])
        robots.append(r)

    return robots


def shortest_path_length(grid, start, goal):
    """
    Compute the shortest path length from start to goal in a grid using A* algorithm.
    start, goal: (x, y)
    Returns length of shortest path, or None if no path.
    """

    if start == goal:
        return 0

    height, width = grid.shape
    open_set = []
    heappush(open_set, (0 + heuristic(start, goal), 0, start[0], start[1]))  # (f_score, g_score, x, y)
    g_score = { (start[0], start[1]): 0 }
    closed_set = set()

    directions = [ (0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1) ]  # Include diagonals

    while open_set:
        f_score, g, x, y = heappop(open_set)

        if (x, y) == goal:
            return g

        closed_set.add((x, y))

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width and grid[nx][ny] == 0:
                if (nx, ny) in closed_set:
                    continue
                tentative_g_score = g + np.hypot(dx, dy)
                if (nx, ny) not in g_score or tentative_g_score < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g_score
                    f = tentative_g_score + heuristic( (nx, ny), goal )
                    heappush(open_set, (f, tentative_g_score, nx, ny))

    return None

def heuristic(a, b):
    # Using Euclidean distance as heuristic
    return np.hypot(a[0]-b[0], a[1]-b[1])



def run_tests(num_tests=10, 
              width=20, 
              height=20, 
              num_robots=5, 
              obstacle_ratio=0.2, 
              max_steps=100):
    """
    Run multiple MAPF test scenarios, collect statistics.
    For each scenario:
      - generate random map
      - place robots
      - run simulation
      - compute optimality metrics
    Print summary statistics at the end.
    """

    optimality_ratios = []  # actual_length / shortest_possible_length for each robot
    success_counts = 0
    total_robots = 0
    times = []

    min_avg_optimality = None
    best_scenario = None

    for test_id in range(num_tests):
        # Generate scenario
        grid = generate_random_map(width, height, obstacle_ratio=obstacle_ratio)
        robots = generate_random_robots(grid, num_robots)

        # Run simulation
        start_time = time.time()
        final_paths, reached_goals = simulate_scenario(grid, robots, max_steps=max_steps, show_path=False, logging=False)
        end_time = time.time()

        # Compute metrics
        scenario_ratios = []
        for i, robot in enumerate(robots):
            actual_path_length = len(final_paths[i]) - 1
            sp = shortest_path_length(grid, (robot.path[0][0], robot.path[0][1]), (robot.goal_x, robot.goal_y))
            if sp is not None and sp > 0:  # Avoid division by zero, sp=0 means start=goal
                ratio = actual_path_length / sp
                scenario_ratios.append(ratio)
                # print (f"Robot {i}: Actual Length: {actual_path_length}, Shortest Path Length: {sp}, Ratio: {ratio:.2f}")
            # If sp is None, no feasible path even ignoring robots. Actual ratio doesn't make sense here.

        # Count how many reached goal
        scenario_success = sum(reached_goals)
        success_counts += scenario_success
        total_robots += num_robots

        if scenario_ratios:
            optimality_ratios.extend(scenario_ratios)
            avg_optimality = np.mean(scenario_ratios)
            # Keep track of scenario with smallest average optimality ratio
            if min_avg_optimality is None or avg_optimality < min_avg_optimality:
                min_avg_optimality = avg_optimality
                best_scenario = (grid, robots, final_paths)

        times.append(end_time - start_time)
        print(f"Test {test_id+1}/{num_tests}:")
        print(f"  Success: {scenario_success}/{num_robots}")
        if scenario_ratios:
            print(f"  Average Optimality Ratio: {np.mean(scenario_ratios):.2f}")
        else:
            print("  No feasible paths to compare optimality.")

    # Print summary statistics
    overall_success_rate = success_counts / (total_robots) if total_robots > 0 else 0
    avg_optimality = np.mean(optimality_ratios) if optimality_ratios else None
    avg_time = np.mean(times) if times else None

    print("\n=== Summary of All Tests ===")
    print(f"Number of tests: {num_tests}")
    print(f"Map Size: {width}x{height}, Robots: {num_robots}, Obstacle Ratio: {obstacle_ratio}")
    print(f"Success Rate (robots reaching goals): {overall_success_rate*100:.2f}%")
    if avg_optimality is not None:
        print(f"Average Optimality Ratio (actual/shortest): {avg_optimality:.2f}")
    else:
        print("No optimality ratio computed (no feasible paths found).")
    if avg_time is not None:
        print(f"Average Computation Time per Scenario: {avg_time:.2f} seconds")

    # After testing, show the map with smallest average optimality ratio
    if best_scenario:
        grid, robots, final_paths = best_scenario
        print("\n=== Map with Smallest Average Optimality Ratio ===")
        print(f"Average Optimality Ratio: {min_avg_optimality:.2f}")
        plot_map_with_paths(grid, robots, final_paths)

if __name__ == "__main__":
    run_tests(num_tests=100,
              width=20,
              height=20,
              num_robots=5,
              obstacle_ratio=0.3,
              max_steps=100)
