import heapq
import math


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
        self.stuck_count = 0
        self.reserved_positions = {}

    def move_towards_goal(self, grid, other_robots):
        self.grid = grid

        # Check if robot is already at the goal
        if (self.x, self.y) == (self.goal_x, self.goal_y):
            return True

        if self.waiting:
            self.waiting = False
            return False

        # Reserve current position
        self.reserved_positions[(self.x, self.y)] = self

        # Attempt to find a path
        path = self.a_star_pathfinding((self.x, self.y), (self.goal_x, self.goal_y), other_robots)

        if not path:
            self.stuck_count += 1
            if self.stuck_count > 3:
                # Make random adjustments to resolve deadlock
                possible_moves = self.get_possible_moves(other_robots)
                if possible_moves:
                    next_position = possible_moves[0]
                    self.x, self.y = next_position
                    self.path.append((self.x, self.y))
                    return True
                self.waiting = True
                return False
        else:
            self.stuck_count = 0

        # Move along the path if it's valid
        if path and len(path) > 1:
            next_position = path[1]
            if self.can_move_to(next_position, other_robots):
                self.x, self.y = next_position
                self.path.append((self.x, self.y))
                self.reserved_positions[next_position] = self
                return True
            elif self.should_wait(next_position, other_robots):
                self.waiting = True
                return False

        return False

    def can_move_to(self, position, other_robots):
        """Check if the robot can move to the next position."""
        return not self.check_for_collisions(position[0], position[1], other_robots)

    def should_wait(self, next_position, other_robots):
        """Determine if the robot should wait based on shared path reservations."""
        for other in other_robots:
            if other is not self:
                if (other.x, other.y) == next_position or next_position in other.reserved_positions:
                    # Prioritize robot closer to the goal
                    my_dist = abs(self.x - self.goal_x) + abs(self.y - self.goal_y)
                    other_dist = abs(other.x - other.goal_x) + abs(other.y - other.goal_y)
                    if my_dist > other_dist:
                        return True
        return False

    def get_possible_moves(self, other_robots):
        """Find all possible moves from the current position."""
        moves = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            if (0 <= new_x < len(self.grid) and
                0 <= new_y < len(self.grid[0]) and
                self.grid[new_x][new_y] == 0 and
                not self.check_for_collisions(new_x, new_y, other_robots)):
                moves.append((new_x, new_y))
        return moves

    def a_star_pathfinding(self, start, goal, other_robots):
        def heuristic(a, b):
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: heuristic(start, goal)}
        oheap = []

        heapq.heappush(oheap, (fscore[start], start))

        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            close_set.add(current)

            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j

                if not (0 <= neighbor[0] < len(self.grid) and 0 <= neighbor[1] < len(self.grid[0])):
                    continue

                if self.grid[neighbor[0]][neighbor[1]] == 1:
                    continue

                if self.check_for_collisions(neighbor[0], neighbor[1], other_robots):
                    continue

                movement_cost = math.sqrt(2) if (i != 0 and j != 0) else 1
                tentative_g_score = gscore[current] + movement_cost

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                    continue

                if tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))

        return None

    def check_for_collisions(self, x, y, other_robots, safety_distance=1.0):
        for other in other_robots:
            if other is not self:
                dist = math.sqrt((other.x - x) ** 2 + (other.y - y) ** 2)
                if dist < safety_distance:
                    return True
                for future_pos in other.path[-3:]:
                    dist = math.sqrt((future_pos[0] - x) ** 2 + (future_pos[1] - y) ** 2)
                    if dist < safety_distance:
                        return True
        return False
