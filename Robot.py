import heapq

class Robot:
    # Shared reservation table across all Robot instances.
    # Keys are (x, y, t) tuples; values are the Robot instance that reserved the cell.
    reservation_table = {}

    def __init__(self, start_x, start_y, goal_x, goal_y, speed=1, start_time=0, deadline=None):
        """
        Initialize a Robot with start and goal positions, speed, start_time, and deadline.
        """
        self.start_x = start_x
        self.start_y = start_y
        self.x = start_x  # current x position
        self.y = start_y  # current y position
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.speed = speed  # assumed to be 1 (one cell per timestep)
        self.start_time = start_time  # simulation time when the robot is allowed to move
        self.deadline = deadline  # latest time by which the goal must be reached (None means no deadline)
        self.path = []          # list of (x, y, t) tuples representing the planned path
        self.path_index = 0     # index into self.path for the next move
        self.at_goal_flag = False   # True if the robot has reached its goal
        self.deadline_missed = False  # True if the robot cannot reach its goal by the deadline

    def get_priority(self, current_time):
        """
        Compute a composite priority tuple for this robot given the current simulation time.
        Lower values indicate higher priority.

        The tuple is defined as:
            (feasible, slack, distance)
        where:
            - feasible is 0 if the remaining time (deadline - current_time) is at least
              the Manhattan distance to the goal (i.e. the robot is feasibly on time),
              and 1 otherwise.
            - slack is (deadline - current_time) - (Manhattan distance), so lower slack means more urgency.
            - distance is the Manhattan distance from the current position to the goal.

        Robots without a deadline are treated as less urgent.
        """
        if self.deadline is None:
            return (1, float('inf'), float('inf'))
        remaining_time = self.deadline - current_time
        distance = abs(self.x - self.goal_x) + abs(self.y - self.goal_y)
        feasible = 0 if remaining_time >= distance else 1
        slack = remaining_time - distance
        return (feasible, slack, distance)

    def at_goal(self):
        """
        Return True if the robot has reached its goal.
        """
        return self.at_goal_flag

    def remove_reservations(self):
        """
        Remove all reservations from the shared reservation table that belong to this robot.
        Useful when re-planning.
        """
        to_remove = [key for key, robot in Robot.reservation_table.items() if robot is self]
        for key in to_remove:
            del Robot.reservation_table[key]

    def plan_path(self, grid, current_time):
        """
        Plan a collision-free path from the robot's current position (self.x, self.y) to its goal
        using a time-extended A* search on the given grid (4-directional movement).

        grid: 2D numpy array where 0 indicates free and 1 indicates an obstacle.
        current_time: simulation time from which planning starts.

        Returns a list of (x, y, t) states if a valid path is found (and reserves these cells),
        or None if no such path exists (within the deadline, if set).
        """
        rows, cols = grid.shape

        def heuristic(x, y):
            return abs(x - self.goal_x) + abs(y - self.goal_y)

        # If already at the goal, simply return the current state.
        if self.x == self.goal_x and self.y == self.goal_y:
            return [(self.x, self.y, current_time)]

        self.remove_reservations()

        open_list = []
        start_state = (self.x, self.y, current_time)
        heapq.heappush(open_list, (heuristic(self.x, self.y), 0, start_state, None))
        came_from = {}
        closed = set()
        found_goal = None

        while open_list:
            f, g, current_state, parent = heapq.heappop(open_list)
            if current_state in closed:
                continue
            closed.add(current_state)
            came_from[current_state] = parent
            x, y, t = current_state

            if x == self.goal_x and y == self.goal_y:
                if self.deadline is None or t <= self.deadline:
                    found_goal = current_state
                    break

            if self.deadline is not None and t >= self.deadline:
                continue

            # 4-directional moves plus waiting
            for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                nt = t + 1
                if not (0 <= nx < rows and 0 <= ny < cols):
                    continue
                if grid[nx, ny] == 1:
                    continue
                if (nx, ny, nt) in Robot.reservation_table and Robot.reservation_table[(nx, ny, nt)] is not self:
                    continue
                # Optional: check swap conflicts.
                if (nx, ny, t) in Robot.reservation_table:
                    other = Robot.reservation_table[(nx, ny, t)]
                    if (x, y, nt) in Robot.reservation_table and Robot.reservation_table[(x, y, nt)] is other:
                        continue

                neighbor_state = (nx, ny, nt)
                if neighbor_state in closed:
                    continue
                new_g = g + 1
                new_f = new_g + heuristic(nx, ny)
                heapq.heappush(open_list, (new_f, new_g, neighbor_state, current_state))

        if found_goal is None:
            return None

        path = []
        state = found_goal
        while state is not None:
            path.append(state)
            state = came_from.get(state)
        path.reverse()

        for state in path:
            Robot.reservation_table[state] = self

        return path

    def yield_move(self, grid, current_time):
        """
        Improved yield move: Evaluate all adjacent free cells,
        check that each candidate is free for the next few timesteps,
        and choose the one that minimizes the Manhattan distance to the goal.
        """
        rows, cols = grid.shape
        best_candidate = None
        best_score = float('inf')
        lookahead_steps = 3  # Check next 3 timesteps for potential conflicts

        # Evaluate each 4-directional adjacent cell.
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = self.x + dx, self.y + dy
            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if grid[nx, ny] == 1:
                continue

            # Check that candidate cell is free for the next few time steps.
            conflict = False
            for dt in range(1, lookahead_steps + 1):
                if (nx, ny, current_time + dt) in Robot.reservation_table:
                    conflict = True
                    break
            if conflict:
                continue

            # Score candidate: lower Manhattan distance means less detour.
            score = abs(nx - self.goal_x) + abs(ny - self.goal_y)
            if score < best_score:
                best_score = score
                best_candidate = (nx, ny)

        return best_candidate

    def step(self, grid, robots, current_time):
        """
        Execute one simulation time step.

        Normal behavior:
          - Re-plan if needed (if there is no current valid plan or the next move is off-sync).
          - Execute the next planned move if its time matches current_time.

        Yielding behavior:
          - If the robot has reached its goal (at_goal_flag is True) but is blocking the entrance
            (e.g. in a tunnel with one entrance) and a higher-priority robot nearby needs that cell,
            then attempt to yield by moving to an adjacent free cell.
        """
        if current_time < self.start_time:
            return

        # --- Improved Yielding Logic ---
        if self.at_goal_flag:
            # Check if the goal cell is reserved for the next timestep by another robot.
            reserved = Robot.reservation_table.get((self.goal_x, self.goal_y, current_time + 1), None)
            if reserved is not None and reserved is not self:
                # Yield if the waiting robot has higher priority.
                if reserved.get_priority(current_time + 1) < self.get_priority(current_time + 1):
                    candidate = self.yield_move(grid, current_time)
                    if candidate is not None:
                        self.remove_reservations()
                        self.x, self.y = candidate
                        self.at_goal_flag = False
                        self.path = []
                        return

            # Additionally, check for any nearby robot (adjacent to the goal cell) that might be stalled.
            for r in robots:
                if r is self or r.at_goal() or r.deadline_missed or current_time < r.start_time:
                    continue
                if abs(r.x - self.goal_x) + abs(r.y - self.goal_y) == 1:
                    if r.get_priority(current_time) < self.get_priority(current_time):
                        candidate = self.yield_move(grid, current_time)
                        if candidate is not None:
                            self.remove_reservations()
                            self.x, self.y = candidate
                            self.at_goal_flag = False
                            self.path = []
                            return

        # If already finished or has failed (after any yielding), do nothing.
        if self.at_goal_flag or self.deadline_missed:
            return

        # Re-plan if necessary (if no path exists or the next move is off-sync).
        if (not self.path or 
            self.path_index >= len(self.path) or 
            self.path[self.path_index][2] != current_time):
            self.remove_reservations()
            new_path = self.plan_path(grid, current_time)
            if new_path is None:
                self.deadline_missed = True
                return
            else:
                self.path = new_path
                self.path_index = 0

        next_state = self.path[self.path_index]
        if next_state[2] == current_time:
            # Check for dynamic conflicts.
            if ((next_state[0], next_state[1], current_time) in Robot.reservation_table and 
                Robot.reservation_table[(next_state[0], next_state[1], current_time)] is not self):
                self.remove_reservations()
                new_path = self.plan_path(grid, current_time)
                if new_path is None:
                    self.deadline_missed = True
                    return
                else:
                    self.path = new_path
                    self.path_index = 0
                    next_state = self.path[self.path_index]

            # Execute the move.
            self.x, self.y, _ = next_state
            self.path_index += 1

            # If the robot reaches its goal, mark it and reserve the cell.
            if self.x == self.goal_x and self.y == self.goal_y:
                self.at_goal_flag = True
                reserve_until = self.deadline if self.deadline is not None else current_time + 500
                for t in range(current_time, reserve_until):
                    Robot.reservation_table[(self.x, self.y, t)] = self
