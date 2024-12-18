import heapq
import numpy as np

class Robot:
    def __init__(self, start_x, start_y, goal_x, goal_y, speed=1):
        self.x = start_x
        self.y = start_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.path = [(self.x, self.y)]  # stores actual visited positions
        self.speed = speed
        self.grid = None
        self.waiting = False
        self.planned_path = []  # Will store tuples of (x, y, t)
        self.id = id(self)

    def calculate_distance_to_goal(self):
        return abs(self.x - self.goal_x) + abs(self.y - self.goal_y)

    def step(self, grid, other_robots, current_time):
        # Perform one step of movement
        self.grid = grid
        if (self.x, self.y) == (self.goal_x, self.goal_y):
            return False  # Already at goal, no move

        # Check if the next move in the current planned path is available
        # If not, or if no planned path, replan
        if not self.has_valid_move_for_time(current_time):
            # Need to plan a path from current position and current_time
            new_path = self.cooperative_a_star_pathfinding(
                start=(self.x, self.y),
                goal=(self.goal_x, self.goal_y),
                other_robots=other_robots,
                start_time=current_time
            )
            if new_path is None:
                # No path found, robot waits
                self.waiting = True
                return False
            else:
                self.planned_path = new_path
                self.waiting = False

        # Execute the move for this time step
        move_made = self.execute_move(current_time)
        return move_made

    def has_valid_move_for_time(self, current_time):
        # Check if we have a planned path that covers current_time+1
        # The robot currently is at (x, y) for current_time.
        # The next move should be at current_time + 1
        for (px, py, pt) in self.planned_path:
            if pt == current_time + 1:
                return True
        return False

    def execute_move(self, current_time):
        # Find the node in planned_path that corresponds to current_time+1
        next_state = None
        for (px, py, pt) in self.planned_path:
            if pt == current_time + 1:
                next_state = (px, py, pt)
                break

        if next_state is None:
            # No move found for next time step, wait in place
            return False

        # Move the robot
        self.x, self.y = next_state[0], next_state[1]
        self.path.append((self.x, self.y))
        return True

    def get_future_position(self, t):
        # Return position at time t if in planned path, else last known
        candidates = [ (px, py) for (px, py, pt) in self.planned_path if pt == t ]
        if candidates:
            return candidates[0]
        # If no future position planned at that time, assume staying at goal or current
        if (self.x, self.y) == (self.goal_x, self.goal_y):
            return (self.x, self.y)
        # If not reached goal and no future position found, robot is waiting in place
        return (self.x, self.y)

    def cooperative_a_star_pathfinding(self, start, goal, other_robots, start_time):
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # all directions
        directions = [
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1)
        ]

        # Build a reservation table from other robots' planned paths
        # Reservations: 
        #   - For nodes: (x, y, t)
        #   - For edges: ((x1,y1),(x2,y2), t) means an edge crossing between t and t+1
        reservation_table = set()

        for robot in other_robots:
            if robot is self:
                continue
            # Reserve all known future positions
            for (rx, ry, rt) in robot.planned_path:
                reservation_table.add((rx, ry, rt))
            # Also reserve stationary positions at the end of their path if needed
            if robot.planned_path:
                last_pos = robot.planned_path[-1]
                lx, ly, lt = last_pos
                # Reserve extra time steps for final position to avoid collisions
                for extra_t in range(lt+1, lt+6):
                    reservation_table.add((lx, ly, extra_t))

            # Reserve edges from their path
            for i in range(len(robot.planned_path)-1):
                (x1, y1, t1) = robot.planned_path[i]
                (x2, y2, t2) = robot.planned_path[i+1]
                # Robot moves from (x1, y1) at time t1 to (x2, y2) at time t2
                # Reserve edge in both directions for that time interval
                reservation_table.add(((x1,y1),(x2,y2),t1))
                reservation_table.add(((x2,y2),(x1,y1),t1))

        # A* with time dimension
        # State: (x, y, t)
        # Start state: (start_x, start_y, start_time)
        # Goal: (goal_x, goal_y) at any time t
        start_state = (start[0], start[1], start_time)
        open_set = []
        heapq.heappush(open_set, (heuristic(start, goal), 0, start_state))
        came_from = {}
        g_score = {start_state: 0}

        while open_set:
            _, _, current = heapq.heappop(open_set)
            (cx, cy, ct) = current

            if (cx, cy) == goal:
                # Reconstruct path
                path = []
                cur_key = current
                while cur_key in came_from:
                    path.append(cur_key)
                    cur_key = came_from[cur_key]
                path.append(cur_key)
                path.reverse()
                return path

            # Try waiting in the same cell (if not reserved)
            next_wait_state = (cx, cy, ct+1)
            if (cx, cy, ct+1) not in reservation_table:
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(next_wait_state, float('inf')):
                    came_from[next_wait_state] = current
                    g_score[next_wait_state] = tentative_g
                    f_score = tentative_g + heuristic((cx, cy), goal)
                    heapq.heappush(open_set, (f_score, tentative_g, next_wait_state))

            # Try moving to neighbors
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                nt = ct + 1

                if not (0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1]):
                    continue
                if self.grid[nx][ny] == 1:
                    continue

                # Check reservations for node and edge conflicts
                # Node reservation
                if (nx, ny, nt) in reservation_table:
                    continue
                # Edge reservation
                if ((cx, cy), (nx, ny), ct) in reservation_table:
                    continue

                tentative_g = g_score[current] + 1
                neighbor_state = (nx, ny, nt)
                if tentative_g < g_score.get(neighbor_state, float('inf')):
                    came_from[neighbor_state] = current
                    g_score[neighbor_state] = tentative_g
                    f_score = tentative_g + heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor_state))

        # No path found
        return None
