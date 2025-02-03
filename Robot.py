import heapq
import numpy as np

class Robot:
    def __init__(self, start_x, start_y, goal_x, goal_y, 
                 speed=1, 
                 start_time=0, 
                 deadline=None):
        """
        :param start_x: Starting grid row.
        :param start_y: Starting grid column.
        :param goal_x:  Goal grid row.
        :param goal_y:  Goal grid column.
        :param speed:   Movement speed (cells per step).
        :param start_time: The time step at which this robot becomes active.
        :param deadline: The time by which this robot must reach its goal (inclusive).
        """
        self.x = start_x
        self.y = start_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.path = [(self.x, self.y)]  # stores visited positions
        self.speed = speed
        self.grid = None
        self.waiting = False
        self.planned_path = []  # Will store tuples (x, y, t)
        self.id = id(self)
        self.number = None  # To be set in main.py for consistent logging

        # Scheduling-related attributes.
        self.start_time = start_time
        self.deadline = deadline
        self.deadline_missed = False  # Becomes True if no valid path exists within deadline.
        self.active = False           # Becomes active when current_time >= start_time.

    def at_goal(self):
        return (self.x, self.y) == (self.goal_x, self.goal_y)

    def calculate_distance_to_goal(self):
        return abs(self.x - self.goal_x) + abs(self.y - self.goal_y)

    def step(self, grid, other_robots, current_time):
        """
        Perform one step of movement (or do nothing).
        If the robot cannot plan a path that reaches its goal by its deadline,
        it immediately marks itself as having missed its deadline.
        :param grid: The occupancy grid.
        :param other_robots: List of all robot objects.
        :param current_time: Current simulation time.
        :return: True if a move was made, False otherwise.
        """
        self.grid = grid

        # Activate when current_time >= start_time.
        if not self.active and current_time >= self.start_time:
            self.active = True
        if not self.active:
            return False

        # If the current time is at or past the deadline, mark as missed.
        if self.deadline is not None and current_time >= self.deadline:
            self.deadline_missed = True
            self.waiting = False  # Clear waiting flag.
            return False

        if self.at_goal():
            return False

        # Check if our planned path has a valid move for current_time+1.
        if not self.has_valid_move_for_time(current_time, other_robots):
            new_path = self.cooperative_a_star_pathfinding(
                start=(self.x, self.y),
                goal=(self.goal_x, self.goal_y),
                other_robots=other_robots,
                start_time=current_time
            )
            if new_path is None:
                # No valid path exists that reaches the goal by the deadline.
                self.deadline_missed = True
                return False
            else:
                self.planned_path = new_path
                self.waiting = False

        return self.execute_move(current_time)

    def has_valid_move_for_time(self, current_time, other_robots):
        """
        Checks that the planned path has a move for time current_time+1 and that
        no other robot’s current position equals the target cell.
        This prevents moving into a cell occupied by a stopped (or waiting) robot.
        """
        next_move = None
        for (px, py, pt) in self.planned_path:
            if pt == current_time + 1:
                next_move = (px, py)
                break
        if next_move is None:
            return False

        for robot in other_robots:
            if robot is self:
                continue
            if (robot.x, robot.y) == next_move:
                return False

        return True

    def execute_move(self, current_time):
        """
        Execute the move scheduled for time current_time+1.
        """
        next_state = None
        for (px, py, pt) in self.planned_path:
            if pt == current_time + 1:
                next_state = (px, py, pt)
                break

        if next_state is None:
            return False

        self.x, self.y = next_state[0], next_state[1]
        self.path.append((self.x, self.y))
        return True

    def get_future_position(self, t):
        candidates = [(px, py) for (px, py, pt) in self.planned_path if pt == t]
        if candidates:
            return candidates[0]
        return (self.x, self.y)

    @staticmethod
    def get_priority(robot):
        """
        Returns a priority tuple.
          - Robots that have NOT missed their deadline get a 0; those that have get a 1.
          - Then the deadline (or infinity if None), then start time, then the permanent number.
        Lower tuples mean higher priority.
        (Note: In this version, robots that miss their deadline stop moving.)
        """
        return (0 if not robot.deadline_missed else 1,
                robot.deadline if robot.deadline is not None else float('inf'),
                robot.start_time,
                robot.number if robot.number is not None else float('inf'))

    def cooperative_a_star_pathfinding(self, start, goal, other_robots, start_time):
        """
        A* search in (x, y, t) space with reservations.
        Reservations come from:
         - Every robot that is at its goal or has missed its deadline (i.e. is stopped), and
         - Moving robots with higher priority than self.
        States with t > deadline are not expanded.
        """
        def heuristic(a, b):
            return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        MAX_HORIZON = (self.grid.shape[0] + self.grid.shape[1]) * 4

        reservation_table = set()
        my_priority = Robot.get_priority(self)
        for other in other_robots:
            if other is self:
                continue
            # Treat robots that are at goal or have missed their deadline as static obstacles.
            if other.at_goal() or other.deadline_missed:
                for extra_t in range(start_time, MAX_HORIZON + 1):
                    reservation_table.add((other.x, other.y, extra_t))
                continue
            # For moving robots with higher priority than self, add their reservations.
            if Robot.get_priority(other) < my_priority:
                for (rx, ry, rt) in other.planned_path:
                    reservation_table.add((rx, ry, rt))
                if other.planned_path:
                    lx, ly, lt = other.planned_path[-1]
                    for extra_t in range(lt + 1, MAX_HORIZON + 1):
                        reservation_table.add((lx, ly, extra_t))
                for i in range(len(other.planned_path) - 1):
                    (x1, y1, t1) = other.planned_path[i]
                    (x2, y2, t2) = other.planned_path[i + 1]
                    reservation_table.add(((x1, y1), (x2, y2), t1))
                    reservation_table.add(((x2, y2), (x1, y1), t1))

        start_state = (start[0], start[1], start_time)
        open_set = []
        heapq.heappush(open_set, (heuristic(start, goal), 0, start_state))
        came_from = {}
        g_score = {start_state: 0}

        while open_set:
            _, _, current = heapq.heappop(open_set)
            cx, cy, ct = current

            if ct > MAX_HORIZON:
                return None
            if self.deadline is not None and not self.deadline_missed and ct > self.deadline:
                continue

            if (cx, cy) == goal:
                path = []
                cur_key = current
                while cur_key in came_from:
                    path.append(cur_key)
                    cur_key = came_from[cur_key]
                path.append(cur_key)
                path.reverse()
                return path

            # Consider waiting in place.
            wait_state = (cx, cy, ct + 1)
            if (cx, cy, ct + 1) not in reservation_table:
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(wait_state, float('inf')):
                    came_from[wait_state] = current
                    g_score[wait_state] = tentative_g
                    f_score = tentative_g + heuristic((cx, cy), goal)
                    heapq.heappush(open_set, (f_score, tentative_g, wait_state))

            # Try moving in all directions.
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                nt = ct + 1
                if not (0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1]):
                    continue
                if self.grid[nx][ny] == 1:
                    continue
                if (nx, ny, nt) in reservation_table:
                    continue
                if ((cx, cy), (nx, ny), ct) in reservation_table:
                    continue

                tentative_g = g_score[current] + 1
                neighbor_state = (nx, ny, nt)
                if tentative_g < g_score.get(neighbor_state, float('inf')):
                    came_from[neighbor_state] = current
                    g_score[neighbor_state] = tentative_g
                    f_score = tentative_g + heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor_state))

        return None
