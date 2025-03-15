import heapq
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Set, Any

logging.basicConfig(level=logging.INFO)


class Robot:
    def __init__(self, start_x: int, start_y: int, goal_x: int, goal_y: int,
                 speed: int = 1,
                 start_time: int = 0,
                 deadline: Optional[int] = None) -> None:
        """
        Initialize the robot with starting and goal positions and scheduling parameters.

        :param start_x: Starting grid row.
        :param start_y: Starting grid column.
        :param goal_x:  Goal grid row.
        :param goal_y:  Goal grid column.
        :param speed:   Movement speed (cells per step).
        :param start_time: The time step at which the robot becomes active.
        :param deadline: The time by which the robot must reach its goal (inclusive).
        """
        self.x: int = start_x
        self.y: int = start_y
        self.goal_x: int = goal_x
        self.goal_y: int = goal_y
        self.path: List[Tuple[int, int]] = [(self.x, self.y)]
        self.speed: int = speed
        self.grid: Optional[np.ndarray] = None
        self.waiting: bool = False
        self.planned_path: List[Tuple[int, int, int]] = []  # Each tuple: (x, y, t)
        self.id: int = id(self)
        self.number: Optional[int] = None  # To be set externally for consistent logging

        # Scheduling-related attributes.
        self.start_time: int = start_time
        self.deadline: Optional[int] = deadline
        self.deadline_missed: bool = False
        self.active: bool = False  # Becomes active when current_time >= start_time

        # For edge collision detection:
        self.last_position: Tuple[int, int] = (start_x, start_y)  # Position before the last move.
        self.last_move_time: Optional[int] = None  # Time when the last move was executed.

    def at_goal(self) -> bool:
        """Returns True if the robot has reached its goal."""
        return (self.x, self.y) == (self.goal_x, self.goal_y)

    def calculate_distance_to_goal(self) -> int:
        """
        Calculates the minimum number of steps required to reach the goal,
        assuming diagonal movement is allowed (Chebyshev distance).
        """
        dx = abs(self.x - self.goal_x)
        dy = abs(self.y - self.goal_y)
        return max(dx, dy)

    def step(self, grid: np.ndarray, other_robots: List["Robot"], current_time: int) -> bool:
        """
        Performs one movement step if possible.
        
        :param grid: The environment grid.
        :param other_robots: List of all other robots (for collision checking).
        :param current_time: The current time step.
        :return: True if a move was executed; False otherwise.
        """
        self.grid = grid

        # Activate the robot if the current time exceeds the start time.
        if not self.active and current_time >= self.start_time:
            self.active = True
        if not self.active or self.at_goal():
            return False

        # Deadline check.
        if self.deadline is not None:
            min_steps_needed = self.calculate_distance_to_goal()
            if current_time + min_steps_needed > self.deadline:
                self.deadline_missed = True
                logging.info(f"Robot {self.number} missed its deadline (earliest arrival > deadline).")
                return False

        # Validate that a planned move exists and is collision-free.
        if not self.has_valid_move_for_time(current_time, other_robots):
            new_path = self.cooperative_a_star_pathfinding(
                start=(self.x, self.y),
                goal=(self.goal_x, self.goal_y),
                other_robots=other_robots,
                start_time=current_time
            )
            if new_path is None:
                self.deadline_missed = True
                logging.info(f"Robot {self.number} could not find a valid path and missed its deadline.")
                return False
            else:
                self.planned_path = new_path
                self.waiting = False

        return self.execute_move(current_time)

    def has_valid_move_for_time(self, current_time: int, other_robots: List["Robot"]) -> bool:
        """
        Checks whether a valid move is scheduled for time current_time+1 that does not result
        in vertex or edge collisions with other robots.
        """
        next_time = current_time + 1
        next_move = next(((px, py) for (px, py, pt) in self.planned_path if pt == next_time), None)
        if next_move is None:
            return False

        my_current_pos = (self.x, self.y)
        for robot in other_robots:
            if robot is self:
                continue

            their_next_pos = robot.get_future_position(next_time)
            their_current_pos = robot.get_future_position(current_time)

            # Vertex collision: the other robot will be in the target cell.
            if their_next_pos == next_move:
                return False

            # Edge collision (swap): the other robot is moving from the target to the current cell.
            if their_next_pos == my_current_pos and their_current_pos == next_move:
                return False

        return True

    def execute_move(self, current_time: int) -> bool:
        """
        Executes the scheduled move for time current_time+1, updates the robot's state,
        and prunes the planned path.
        """
        next_time = current_time + 1
        next_state = next(((px, py, pt) for (px, py, pt) in self.planned_path if pt == next_time), None)
        if next_state is None:
            return False

        # Record the previous position for collision detection.
        self.last_position = (self.x, self.y)
        # Update to the new position.
        self.x, self.y = next_state[0], next_state[1]
        self.path.append((self.x, self.y))
        self.last_move_time = next_time
        # Remove moves that have been executed.
        self.planned_path = [step for step in self.planned_path if step[2] > next_time]
        return True

    def get_future_position(self, t: int) -> Tuple[int, int]:
        """
        Returns the predicted position of the robot at time t.
        
        :param t: The time step to query.
        :return: A tuple (x, y) representing the position.
        """
        if self.last_move_time is not None and t == self.last_move_time - 1:
            return self.last_position
        candidate = next(((px, py) for (px, py, pt) in self.planned_path if pt == t), None)
        return candidate if candidate is not None else (self.x, self.y)

    @staticmethod
    def get_priority(robot: "Robot") -> Tuple:
        """
        Computes a priority tuple for the robot. Lower tuple values indicate higher priority.
        
        Priority is determined by:
         1) Whether the robot has missed its deadline (0 if not, else 1).
         2) The robot's deadline (or inf if None).
         3) The robot's start time.
         4) The Chebyshev distance from the goal.
         5) The robot number (for tie-breaking).
        """
        remaining_distance = max(abs(robot.x - robot.goal_x), abs(robot.y - robot.goal_y))
        return (
            0 if not robot.deadline_missed else 1,
            robot.deadline if robot.deadline is not None else float('inf'),
            robot.start_time,
            remaining_distance,
            robot.number if robot.number is not None else float('inf')
        )

    @staticmethod
    def move_cost(dx: int, dy: int) -> float:
        """
        Returns the cost of moving given a delta.
        Diagonal moves cost sqrt(2); cardinal moves cost 1.
        """
        return np.sqrt(2) if dx != 0 and dy != 0 else 1

    @staticmethod
    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Diagonal (octile) distance heuristic for A*.
        """
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)

    def build_reservation_table(
        self, other_robots: List["Robot"], start_time: int, MAX_HORIZON: int
    ) -> Dict[int, Dict[str, Set[Any]]]:
        """
        Builds a reservation table (a mapping from time steps to reserved vertices and edges)
        to avoid collisions during planning.
        """
        reservation_table = {
            t: {'vertices': set(), 'edges': set()} for t in range(start_time, MAX_HORIZON + 1)
        }
        my_priority = Robot.get_priority(self)
        for other in other_robots:
            if other is self:
                continue

            # If the other robot is static, reserve its goal position for all future time steps.
            if other.at_goal() or other.deadline_missed:
                for t in range(start_time, MAX_HORIZON + 1):
                    reservation_table[t]['vertices'].add((other.x, other.y))
                continue

            # Reserve positions and edges for robots with higher or equal priority.
            if Robot.get_priority(other) <= my_priority:
                for (rx, ry, rt) in other.planned_path:
                    if rt in reservation_table:
                        reservation_table[rt]['vertices'].add((rx, ry))
                if other.planned_path:
                    lx, ly, lt = other.planned_path[-1]
                    for t in range(lt + 1, MAX_HORIZON + 1):
                        reservation_table[t]['vertices'].add((lx, ly))
                # Reserve edges to prevent swap conflicts.
                for i in range(len(other.planned_path) - 1):
                    (x1, y1, t1) = other.planned_path[i]
                    (x2, y2, t2) = other.planned_path[i + 1]
                    if t1 in reservation_table:
                        reservation_table[t1]['edges'].add(((x1, y1), (x2, y2)))
                        reservation_table[t1]['edges'].add(((x2, y2), (x1, y1)))
        return reservation_table

    def reconstruct_path(
        self, came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]], 
        current: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int]]:
        """
        Reconstructs the path from the came_from dictionary after reaching the goal.
        """
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        return list(reversed(path))

    def cooperative_a_star_pathfinding(
        self, start: Tuple[int, int], goal: Tuple[int, int],
        other_robots: List["Robot"], start_time: int
    ) -> Optional[List[Tuple[int, int, int]]]:
        """
        Performs cooperative A* search in (x, y, t) space using a reservation table to
        avoid conflicts with other robots.
        """
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        max_rows, max_cols = self.grid.shape
        MAX_HORIZON = (max_rows + max_cols) * 4
        reservation_table = self.build_reservation_table(other_robots, start_time, MAX_HORIZON)

        start_state = (start[0], start[1], start_time)
        open_set = []
        heapq.heappush(open_set, (Robot.heuristic(start, goal), 0, start_state))
        came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        g_score: Dict[Tuple[int, int, int], float] = {start_state: 0}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            cx, cy, ct = current

            if ct > MAX_HORIZON:
                logging.info(f"Robot {self.number} search exceeded maximum horizon.")
                return None
            if self.deadline is not None and ct > self.deadline:
                continue

            if (cx, cy) == goal:
                return self.reconstruct_path(came_from, current)

            next_t = ct + 1
            # Option to wait in place.
            if self.deadline is None or next_t <= self.deadline:
                if (cx, cy) not in reservation_table[next_t]['vertices']:
                    wait_state = (cx, cy, next_t)
                    tentative_g = current_g + 1  # Waiting cost.
                    if tentative_g < g_score.get(wait_state, float('inf')):
                        came_from[wait_state] = current
                        g_score[wait_state] = tentative_g
                        f_score = tentative_g + Robot.heuristic((cx, cy), goal)
                        heapq.heappush(open_set, (f_score, tentative_g, wait_state))

            # Try moving in all allowed directions.
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                nt = ct + 1
                if self.deadline is not None and nt > self.deadline:
                    continue
                if not (0 <= nx < max_rows and 0 <= ny < max_cols):
                    continue
                if self.grid[nx][ny] == 1:
                    continue
                if (nx, ny) in reservation_table[nt]['vertices']:
                    continue
                if ((cx, cy), (nx, ny)) in reservation_table[ct]['edges']:
                    continue

                cost = Robot.move_cost(dx, dy)
                tentative_g = current_g + cost
                neighbor_state = (nx, ny, nt)
                if tentative_g < g_score.get(neighbor_state, float('inf')):
                    came_from[neighbor_state] = current
                    g_score[neighbor_state] = tentative_g
                    f_score = tentative_g + Robot.heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor_state))

        logging.info(f"Robot {self.number} failed to find a path.")
        return None
