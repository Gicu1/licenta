import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.ticker import MaxNLocator
import os
import shutil
from Robot import Robot

def run_simulation():
    # Initialize output directories
    output_dir = "C:/school/programare/LICENTA/licenta"
    results_dir = os.path.join(output_dir, "results")
    temp_dir = os.path.join(output_dir, "temp")

    # Clear temp directory if it exists
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    for dir_path in [output_dir, results_dir, temp_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Map dimensions
    map_width = 20
    map_height = 20

    # Initialize grid
    grid = np.zeros((map_height, map_width), dtype=int)

    # Add obstacles
    obstacles = [
        (5, 4), (5, 5), (5, 6), (5, 7), (15, 15), (15, 16),(2,9),(2,11),
        (3,9),(3,11),(4,9),(4,11),(5,9),(5,11),(6,9),(6,11),(7,9),(7,11),(8,9),(8,11),
        (10,9),(10,11),(12,9),(12,11),(14,9),(14,11),(15,9),(15,11),(16,9),(16,11),
        (8,12),(9,12),(10,12),(8,8),(9,8),(10,8),(5,8),(8,13),(8,14),(8,15),
        (9,9), (9,11),
        (12,8),(14,8),(12,12),(14,12), (13,8), (13,12), (8,16), (8,17), (8,18), (8,19), (5,3), (5,2), (5,1),
        (3,0), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7),
        (7,0), (7,1), (7,2), (7,3), (7,4), (7,5), (7,6),
        (9,1), (9,2), (9,3), (9,4), (9,5), (9,6), (9,7),
        (11,0), (11,1), (11,2), (11,3), (11,4), (11,5), (11,6),
        (13,9), (13,11), (11,8), (11,12),
        # (11,9),(11,11),
        # (3,8),(3,10),
    ]
    for (y, x) in obstacles:
        grid[y][x] = 1

    # Initialize robots
    robots = [
        Robot(0, 10, 18, 10),
        Robot(18, 9, 1, 11),
        Robot(18, 0, 0, 18),
        Robot(0, 18, 18, 0),
        Robot(10, 13, 18, 18),
        Robot(6, 12, 6, 12),
    ]

    # Store images for GIF creation
    images = []
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

    # Variable to control whether to show the whole path or just the current position
    show_path = False

    # Simulation steps
    max_steps = 100
    log_path = os.path.join(results_dir, 'simulation_log.txt')
    with open(log_path, 'w') as log_file:
        for step in range(max_steps):
            log_file.write(f"\nStep {step+1}\n")
            moves_made = False

            # Move each robot
            for i, robot in enumerate(robots):
                moved = robot.step(grid, robots, step)
                moves_made = moves_made or moved
                log_file.write(f"Robot {i} position: ({robot.x}, {robot.y})\n")

            # Create and save current map state
            plt.figure(figsize=(10, 10))
            plt.imshow(grid, cmap='binary', interpolation='nearest')

            # Plot robot paths or current positions
            for i, robot in enumerate(robots):
                if show_path and robot.path:
                    plt.plot([pos[1] for pos in robot.path], [pos[0] for pos in robot.path],
                             color=colors[i % len(colors)], marker='o', label=f"Robot {i+1}")
                else:
                    plt.plot(robot.y, robot.x, color=colors[i % len(colors)], marker='o', label=f"Robot {i+1}")

            plt.title(f"Step {step+1}")
            plt.legend(loc='upper right')
            plt.grid(True)

            # Set integer ticks
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

            # Save frame
            buf = os.path.join(temp_dir, f"step_{step:03d}.png")
            plt.savefig(buf)
            images.append(Image.open(buf))
            plt.close()

            # Check if all robots reached their goals
            if all((robot.x, robot.y) == (robot.goal_x, robot.goal_y) for robot in robots):
                log_file.write("All robots reached their goals!\n")
                break

            # Check if no moves were made
            if not moves_made:
                log_file.write("No moves possible - simulation stuck\n")
                break

    # Create GIF
    gif_path = os.path.join(results_dir, 'robot_simulation.gif')
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=200,
        loop=0
    )

    print(f"Simulation complete. GIF saved to {gif_path}")

if __name__ == "__main__":
    run_simulation()
