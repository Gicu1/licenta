import heapq
import numpy as np

class Robot:
    def __init__(self, start_x, start_y, goal_x, goal_y, speed=1):
        self.x = start_x
        self.y = start_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.path = [(self.x, self.y)]
        self.speed = speed
        self.grid = None
        self.waiting = False
        self.planned_path = []
        self.id = id(self)

    def calculate_distance_to_goal(self):
        return abs(self.x - self.goal_x) + abs(self.y - self.goal_y)

    def move_towards_goal(self, grid, other_robots, current_time):
        self.grid = grid

        if (self.x, self.y) == (self.goal_x, self.goal_y):
            return False  # Already at the goal

        if not self.planned_path or self.path_conflict(other_robots, current_time):
            path = self.cooperative_a_star_pathfinding(
                (self.x, self.y),
                (self.goal_x, self.goal_y),
                other_robots,
                current_time
            )
            if path:
                self.planned_path = path
            else:
                self.waiting = True
                return False  # Cannot find a path, wait

        if len(self.planned_path) > 1:
            next_position = self.planned_path[1]
            self.x, self.y = next_position
            self.path.append((self.x, self.y))
            self.planned_path.pop(0)
            return True
        else:
            return False  # Path is empty, wait

    def path_conflict(self, other_robots, current_time):
        for robot in other_robots:
            if robot is self:
                continue
            future_pos = robot.get_future_position(current_time + 1)
            if future_pos == (self.x, self.y):
                return True
        return False

    def cooperative_a_star_pathfinding(self, start, goal, other_robots, current_time):
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        directions = [
            (0, 1),  (1, 0),  (0, -1), (-1, 0),
            (1, 1),  (-1, -1), (1, -1), (-1, 1)
        ]

        # Build reservation table with vertex and edge collisions
        reservation_table = {}
        for robot in other_robots:
            if robot is self:
                continue
            time = current_time
            prev_pos = None
            for pos in robot.planned_path:
                reservation_table.setdefault((pos, time), robot.id)
                if prev_pos is not None:
                    edge = (prev_pos, pos)
                    reservation_table.setdefault((edge, time - 1), robot.id)
                prev_pos = pos
                time += 1
            # Reserve the goal position for other robots after they arrive
            if robot.planned_path:
                final_pos = robot.planned_path[-1]
                for t in range(time, time + 5):  # Reserve for additional steps
                    reservation_table.setdefault((final_pos, t), robot.id)

        open_set = []
        heapq.heappush(open_set, (heuristic(start, goal), current_time, start))
        came_from = {}
        g_score = {(start, current_time): 0}

        while open_set:
            _, time_step, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                key = (current, time_step)
                while key in came_from:
                    key = came_from[key]
                    path.append(key[0])
                path.reverse()
                return path

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                next_time_step = time_step + 1

                if not (0 <= neighbor[0] < self.grid.shape[0] and 0 <= neighbor[1] < self.grid.shape[1]):
                    continue  # Out of bounds
                if self.grid[neighbor[0]][neighbor[1]] == 1:
                    continue  # Obstacle

                # Check for vertex collision
                if reservation_table.get((neighbor, next_time_step)) is not None:
                    continue

                # Check for edge collision
                edge = (current, neighbor)
                reverse_edge = (neighbor, current)
                if reservation_table.get((edge, time_step)) is not None or reservation_table.get((reverse_edge, time_step)) is not None:
                    continue

                tentative_g_score = g_score.get((current, time_step), float('inf')) + 1
                neighbor_key = (neighbor, next_time_step)

                if tentative_g_score < g_score.get(neighbor_key, float('inf')):
                    came_from[neighbor_key] = (current, time_step)
                    g_score[neighbor_key] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, next_time_step, neighbor))

            # Option to wait in current position
            next_time_step = time_step + 1
            if reservation_table.get((current, next_time_step)) is None:
                tentative_g_score = g_score.get((current, time_step), float('inf')) + 1
                neighbor_key = (current, next_time_step)
                if tentative_g_score < g_score.get(neighbor_key, float('inf')):
                    came_from[neighbor_key] = (current, time_step)
                    g_score[neighbor_key] = tentative_g_score
                    f_score = tentative_g_score + heuristic(current, goal)
                    heapq.heappush(open_set, (f_score, next_time_step, current))

        return None  # No path found

    def get_future_position(self, t):
        if t < len(self.planned_path):
            return self.planned_path[t]
        elif self.planned_path:
            return self.planned_path[-1]
        else:
            return (self.x, self.y)