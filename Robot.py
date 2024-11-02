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

    def move_towards_goal(self, grid, other_robots, max_retries=5):
        self.grid = grid

        # Check if robot is already at the goal
        if (self.x, self.y) == (self.goal_x, self.goal_y):
            return True

        if self.waiting:
            self.waiting = False
            return False

        # Try reaching the goal or a reachable location
        path = self.a_star_pathfinding((self.x, self.y), (self.goal_x, self.goal_y), other_robots)
        
        if not path:
            self.stuck_count += 1
            if self.stuck_count > 3:
                path = self.find_nearest_reachable_goal(other_robots)
                if path:
                    self.stuck_count = 0
                else:
                    # Try to make a small random move to break deadlock
                    possible_moves = self.get_possible_moves(other_robots)
                    if possible_moves:
                        next_position = possible_moves[0]  # Take first available move
                        self.x, self.y = next_position
                        self.path.append((self.x, self.y))
                        return True
                    self.waiting = True
                    return False
        else:
            self.stuck_count = 0

        if path and len(path) > 1:
            next_position = path[1]
            if not self.check_for_collisions(next_position[0], next_position[1], other_robots):
                self.x, self.y = next_position
                self.path.append((self.x, self.y))
                return True

        return False

    def get_possible_moves(self, other_robots):
        """Get all possible moves from current position"""
        moves = []
        # Include diagonal moves
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            new_x, new_y = self.x + dx, self.y + dy
            
            if (0 <= new_x < len(self.grid) and 
                0 <= new_y < len(self.grid[0]) and 
                self.grid[new_x][new_y] == 0 and
                not self.check_for_collisions(new_x, new_y, other_robots)):
                
                # Calculate if this move gets us closer to the goal
                current_dist = abs(self.x - self.goal_x) + abs(self.y - self.goal_y)
                new_dist = abs(new_x - self.goal_x) + abs(new_y - self.goal_y)
                
                if new_dist < current_dist:
                    moves.append((new_x, new_y))
        
        return moves

    def a_star_pathfinding(self, start, goal, other_robots):
        def heuristic(a, b):
            # Diagonal distance heuristic
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

        # Include diagonal moves
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

                # Use diagonal distance for movement cost
                movement_cost = math.sqrt(2) if (i != 0 and j != 0) else 1
                tentative_g_score = gscore[current] + movement_cost

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                    continue

                if tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return False

    def find_nearest_reachable_goal(self, other_robots, max_radius=5):
        """Find the nearest reachable goal if the main path is blocked."""
        tried_cells = set()

        for radius in range(1, max_radius + 1):
            # Include diagonal positions in the search
            positions = [(self.x + dx, self.y + dy)
                        for dx in range(-radius, radius + 1)
                        for dy in range(-radius, radius + 1)
                        if (dx, dy) != (0, 0)]
            
            # Sort positions by distance to goal
            positions.sort(key=lambda pos: abs(pos[0] - self.goal_x) + abs(pos[1] - self.goal_y))

            for pos in positions:
                if pos in tried_cells:
                    continue

                tried_cells.add(pos)
                
                if (0 <= pos[0] < len(self.grid) and 
                    0 <= pos[1] < len(self.grid[0]) and
                    self.grid[pos[0]][pos[1]] == 0 and
                    not self.check_for_collisions(pos[0], pos[1], other_robots)):
                    
                    path = self.a_star_pathfinding((self.x, self.y), pos, other_robots)
                    if path:
                        return path

        return None

    def check_for_collisions(self, x, y, other_robots, safety_distance=1.0):
        for other in other_robots:
            if other is not self:
                dist = math.sqrt((other.x - x)**2 + (other.y - y)**2)
                if dist < safety_distance:
                    return True
        return False
