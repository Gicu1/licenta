import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
import time
import random
from heapq import heappush, heappop
from Robot import Robot  # Make sure Robot.py (the new version) is in the same folder


def plot_map_with_paths(grid, robots, final_paths):
    """
    Plot the grid map and overlay the paths of the robots.
    Start positions are marked in green and goals in red.
    """
    plt.figure(figsize=(10, 10))
    height, width = grid.shape
    plt.imshow(grid, cmap='gray_r')
    for i, path in enumerate(final_paths):
        x_coords = [pos[1] for pos in path]
        y_coords = [pos[0] for pos in path]
        plt.plot(x_coords, y_coords, marker='o', label=f'Robot {robots[i].number}')
        plt.scatter(x_coords[0], y_coords[0], c='green', marker='o')  # Start
        plt.scatter(x_coords[-1], y_coords[-1], c='red', marker='x')  # Goal
    plt.title('Map with Robots’ Paths')
    plt.legend()
    plt.show()


def simulate_scenario(grid, robots, max_steps=100, logging=False):
    """
    Run the MAPF simulation on a given grid and set of robots.
    The simulation stops when all robots have reached their goals or have missed their deadlines.
    Returns:
        final_paths: list of robot paths (list of (x,y) positions) for each robot.
        reached_goals: list of booleans indicating if each robot reached its goal.
    """
    grid_copy = np.copy(grid)
    num_robots = len(robots)
    log_lines = []

    for step in range(max_steps):
        if logging:
            log_lines.append(f"\nStep {step + 1}\n")
        moves_made = False

        # Determine active robots (those whose start time has arrived and who have not stopped)
        active_robots = [r for r in robots if step >= r.start_time and not (r.at_goal() or r.deadline_missed)]
        active_robots.sort(key=Robot.get_priority)

        for robot in active_robots:
            moved = robot.step(grid_copy, robots, step)
            moves_made = moves_made or moved
            if logging:
                log_lines.append(f"Robot {robot.number} position: ({robot.x}, {robot.y})\n")

        # If all robots are done (reached goal or missed deadline), exit simulation.
        if all(r.at_goal() or r.deadline_missed for r in robots):
            if logging:
                log_lines.append("All robots reached their goals or missed their deadlines!\n")
            break

        # If no moves were made in this step, we assume the simulation is stuck.
        if not moves_made:
            if logging:
                log_lines.append("No moves possible -- simulation appears to be stuck.\n")
            break

    if logging:
        print("".join(log_lines))

    final_paths = [r.path for r in robots]
    reached_goals = [r.at_goal() for r in robots]
    return final_paths, reached_goals


def generate_random_map(width, height, obstacle_ratio=0.2):
    """
    Generate a random grid (numpy 2D array) where 0 represents a free cell and 1 represents an obstacle.
    """
    grid = np.zeros((height, width), dtype=int)
    num_cells = width * height
    num_obstacles = int(obstacle_ratio * num_cells)
    obstacles = random.sample([(i, j) for i in range(height) for j in range(width)], num_obstacles)
    for (i, j) in obstacles:
        grid[i][j] = 1
    return grid


def heuristic(a, b):
    """
    Use Chebyshev distance as an admissible heuristic (since our robots move in 8 directions,
    and each move costs 1).
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def shortest_path_length(grid, start, goal):
    """
    Compute the shortest path length (in steps) from start to goal on the grid using A*.
    Each move (in any of the 8 directions) costs 1.
    Returns the number of steps (integer) of the shortest path, or None if no path exists.
    """
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


def generate_random_robots(grid, num_robots, buffer_time=10):
    """
    Generate a list of random robots placed on free cells in the grid.
    For each robot, choose distinct free cells for start and goal.
    The deadline for each robot is set to (shortest path length + buffer_time).
    Only robots with a feasible (static) path are generated.
    All robots have start_time = 0.
    """
    height, width = grid.shape
    free_cells = [(i, j) for i in range(height) for j in range(width) if grid[i][j] == 0]
    if len(free_cells) < 2 * num_robots:
        raise ValueError("Not enough free cells to place all robots.")

    robots = []
    trials = 0
    # Keep trying until we have generated the desired number of robots.
    while len(robots) < num_robots and trials < 1000:
        trials += 1
        start, goal = random.sample(free_cells, 2)
        sp = shortest_path_length(grid, start, goal)
        if sp is None:
            continue  # No path exists; try a different pair.
        deadline = int(sp) + buffer_time  # Deadline is reachable if robot follows near-optimal route.
        r = Robot(start_x=start[0],
                  start_y=start[1],
                  goal_x=goal[0],
                  goal_y=goal[1],
                  start_time=0,
                  deadline=deadline)
        robots.append(r)
    if len(robots) < num_robots:
        raise RuntimeError("Could not generate enough valid robots for the scenario.")
    # Assign permanent numbers for logging purposes.
    for i, robot in enumerate(robots):
        robot.number = i + 1
    return robots


def run_tests(num_tests=10, width=20, height=20, num_robots=5, obstacle_ratio=0.3, max_steps=100):
    """
    Run multiple MAPF test scenarios.
    For each scenario:
      - Generate a random map.
      - Place robots (with deadlines that are reachable).
      - Run the simulation.
      - Compute optimality metrics (actual steps taken / shortest path steps).
    Summary statistics are printed at the end.
    """
    optimality_ratios = []  # For each robot: actual_steps / shortest_possible_steps
    success_counts = 0
    total_robots = 0
    times = []

    min_avg_optimality = None
    best_scenario = None

    for test_id in range(num_tests):
        # Generate scenario.
        grid = generate_random_map(width, height, obstacle_ratio=obstacle_ratio)
        robots = generate_random_robots(grid, num_robots, buffer_time=10)

        start_time = time.time()
        final_paths, reached_goals = simulate_scenario(grid, robots, max_steps=max_steps, logging=False)
        end_time = time.time()

        scenario_ratios = []
        for i, robot in enumerate(robots):
            actual_steps = len(final_paths[i]) - 1  # Number of moves taken
            sp = shortest_path_length(grid, (final_paths[i][0][0], final_paths[i][0][1]), (robot.goal_x, robot.goal_y))
            if sp is not None and sp > 0:
                ratio = actual_steps / sp
                scenario_ratios.append(ratio)
            # (If sp is None or 0, we skip ratio computation.)

        scenario_success = sum(reached_goals)
        success_counts += scenario_success
        total_robots += num_robots

        if scenario_ratios:
            avg_opt = np.mean(scenario_ratios)
            optimality_ratios.extend(scenario_ratios)
            if min_avg_optimality is None or avg_opt < min_avg_optimality:
                min_avg_optimality = avg_opt
                best_scenario = (grid, robots, final_paths)
        times.append(end_time - start_time)

        print(f"Test {test_id + 1}/{num_tests}:")
        print(f"  Robots reaching goal: {scenario_success}/{num_robots}")
        if scenario_ratios:
            print(f"  Average Optimality Ratio (actual/shortest): {np.mean(scenario_ratios):.2f}")
        else:
            print("  No optimality ratio computed (no valid paths found).")

    overall_success_rate = (success_counts / total_robots) if total_robots > 0 else 0
    avg_optimality = np.mean(optimality_ratios) if optimality_ratios else None
    avg_time = np.mean(times) if times else None

    print("\n=== Summary ===")
    print(f"Total Tests: {num_tests}")
    print(f"Map Size: {width}x{height}, Robots per scenario: {num_robots}, Obstacle Ratio: {obstacle_ratio}")
    print(f"Overall Success Rate (robots reaching goals): {overall_success_rate * 100:.2f}%")
    if avg_optimality is not None:
        print(f"Overall Average Optimality Ratio (actual/shortest): {avg_optimality:.2f}")
    else:
        print("No optimality ratios computed.")
    if avg_time is not None:
        print(f"Average Computation Time per Scenario: {avg_time:.2f} seconds")

    # Display the best scenario (lowest average optimality ratio)
    if best_scenario is not None:
        grid_best, robots_best, final_paths_best = best_scenario
        print("\n=== Best Scenario (Lowest Average Optimality Ratio) ===")
        print(f"Average Optimality Ratio: {min_avg_optimality:.2f}")
        plot_map_with_paths(grid_best, robots_best, final_paths_best)


if __name__ == "__main__":
    run_tests(
        num_tests=20,
        width=20,
        height=20,
        num_robots=5,
        obstacle_ratio=0.3,
        max_steps=100
    )
