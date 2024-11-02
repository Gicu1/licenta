import numpy as np
import matplotlib.pyplot as plt
from Robot import Robot

# Dimensiunile hărții
map_width = 20
map_height = 20

# Inițializarea hărții ca o matrice 2D, 0 = zonă liberă, 1 = obstacol
grid = np.zeros((map_height, map_width), dtype=int)

# Adăugarea obstacolelor 
obstacles = [(5, 5), (5, 6), (5, 7), (10, 10), (15, 15), (15, 16)]
for (y, x) in obstacles:
    grid[y][x] = 1

# Vizualizarea hărții cu obstacole
# plt.imshow(grid, cmap='binary')
# plt.title("Harta cu Obstacole")
# plt.show()


# Poziția de start și destinația robotului
start_position = (0, 0)
goal_position = (18, 18)

# Inițializarea robotului
robot = Robot(start_position[0], start_position[1], goal_position[0], goal_position[1])

# Simularea mișcării robotului până ajunge la destinație
for _ in range(50):  # 50 pași
    robot.move_towards_goal()

# Plotare hartă și traiectorie robot
plt.imshow(grid, cmap='binary')
plt.plot([pos[1] for pos in robot.path], [pos[0] for pos in robot.path], color='blue', marker='o')
plt.title("Traiectoria Robotului")
plt.show()
