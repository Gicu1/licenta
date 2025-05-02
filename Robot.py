import heapq
import sys
from typing import Dict, List, Tuple, Optional, Set, TYPE_CHECKING, Any
import numpy as np 

# Define type aliases for clarity
Position = Tuple[int, int]
TimeStep = int
State = Tuple[int, int, TimeStep]  # (x, y, t) - Assuming x=row, y=column
Path = List[State]
Grid = np.ndarray

class Robot:
    """
    Represents a robot navigating a grid environment using time-expanded A*
    and a shared reservation table for collision avoidance with prioritized planning.

    Coordinates: Assumes grid accessed as grid[row, col], so state is (row, col, time).

    Attributes:
        reservation_table (Dict[State, 'Robot']): Class-level dictionary storing
            reserved cells at specific timesteps. Key: (row, col, t), Value: Robot instance.
        DIRECTIONS (List[Tuple[int, int]]): Possible movements (dr, dc) including waiting.
            Order: Wait(0,0), Down(1,0), Up(-1,0), Right(0,1), Left(0,-1).
    """
    reservation_table: Dict[State, 'Robot'] = {}
    # Movement deltas (dr, dc) relative to (row, col)
    DIRECTIONS: List[Tuple[int, int]] = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

    def __init__(self,
                 start_x: int,
                 start_y: int,
                 goal_x: int,
                 goal_y: int,
                 robot_id: Any = None,
                 start_time: TimeStep = 0,
                 deadline: Optional[TimeStep] = None):
        """
        Initializes a Robot instance.

        Args:
            start_x (int): Initial row index.
            start_y (int): Initial column index.
            goal_x (int): Target row index.
            goal_y (int): Target column index.
            robot_id (any): An identifier for the robot (e.g., number, string). Defaults to None.
            start_time (TimeStep): Simulation time when the robot can start moving. Defaults to 0.
            deadline (Optional[TimeStep]): Latest simulation time the goal must be reached.
                                            None means no deadline. Defaults to None.
        """

        self.robot_id = robot_id if robot_id is not None else id(self)
        self.start_x: int = start_x
        self.start_y: int = start_y
        self.x: int = start_x  # Current row position
        self.y: int = start_y  # Current column position
        self.goal_x: int = goal_x
        self.goal_y: int = goal_y
        self.start_time: TimeStep = start_time
        self.deadline: Optional[TimeStep] = deadline

        self.path: Path = []          # Planned path: List of (row, col, t) tuples
        self.path_index: int = 0     # Index into self.path corresponding to the state for the *current* time
        self.at_goal_flag: bool = False # True if the robot has reached its goal
        self.deadline_missed: bool = False # True if the robot cannot reach goal by deadline or failed planning
        self.total_moves: int = 0          # how many times we executed an action
        self.arrival_time: Optional[int] = None  # wall‑clock time when goal reached
        
        # Counter for consecutive planning failures to detect stuck robots
        self.consecutive_planning_failures: int = 0
        self.max_planning_failures: int = 5  # Mark as failed after this many consecutive failures

        # Stores the set of (row, col, t) states currently reserved by this robot instance.
        self.my_reservations: Set[State] = set()
        
    def __repr__(self) -> str:
        """Provides a string representation of the robot's current state."""
        status = "Goal" if self.at_goal() else ("Failed" if self.deadline_missed else "Active")
        path_len = len(self.path) - self.path_index if self.path else 0
        return (f"Robot(id={self.robot_id}, pos=({self.x},{self.y}), goal=({self.goal_x},{self.goal_y}), "
                f"S={self.start_time}, D={self.deadline}, status={status}, path_rem={path_len})")

    @classmethod
    def clear_reservations(cls) -> None:
        """
        Clears the shared reservation table.
        MUST be called once before starting each new simulation run.
        """
        cls.reservation_table.clear()

    def get_priority(self, current_time: TimeStep) -> Tuple[int, float, float]:
        """
        Computes priority based primarily on DEADLINE. Lower value = higher priority.

        Priority Tuple: (feasible_code, deadline_value, distance)
            - feasible_code (int): 0=OK, 1=NoDeadline, 2=Infeasible/Failed.
            - deadline_value (float): Actual deadline time. Lower is higher priority. Inf if no deadline.
            - distance (float): Manhattan distance. Tie-breaker. Lower is higher priority.

        Args:
            current_time (TimeStep): The current simulation time.

        Returns:
            Tuple[int, float, float]: The priority tuple.
        """
        if self.at_goal() or self.deadline_missed: # Lowest priority if finished or failed
            # Assign values that sort last
            return (2, float('inf'), float('inf'))

        # Calculate feasibility based on Manhattan distance for code assignment
        distance = self.manhattan_distance_to_goal()
        if self.deadline is None:
            feasible_code = 1 # No deadline code
            deadline_value = float('inf') # Sorts after finite deadlines
            tie_breaker = distance
        else:
            remaining_time = float(self.deadline - current_time)
            if remaining_time >= distance:
                feasible_code = 0 # Feasible deadline code
                deadline_value = float(self.deadline) # Use actual deadline for sorting
                tie_breaker = distance
            else:
                # Infeasible / Deadline missed or will be missed
                feasible_code = 2 # Infeasible code
                deadline_value = float(self.deadline) # Keep deadline for potential sorting among infeasible
                tie_breaker = distance

        # Tuple structure ensures lexicographical comparison:
        # 1. Feasibility (0 < 1 < 2)
        # 2. Deadline (lower deadline first)
        # 3. Distance (lower distance first)
        return (feasible_code, deadline_value, tie_breaker)


    def manhattan_distance_to_goal(self) -> int:
        """Calculates the Manhattan distance from the current position to the goal."""
        return abs(self.x - self.goal_x) + abs(self.y - self.goal_y)

    def heuristic(self, x: int, y: int) -> int:
        """Heuristic function (Manhattan distance) used in A* search."""
        return abs(x - self.goal_x) + abs(y - self.goal_y)

    def at_goal(self) -> bool:
        """Returns True if the robot is currently at its goal position."""
        # Check flag first for performance, then position for correctness
        return self.at_goal_flag and self.x == self.goal_x and self.y == self.goal_y

    def remove_reservations(self) -> None:
        """
        Removes all reservations previously made by *this* robot instance
        from the shared reservation table using its local `my_reservations` set.
        """
        # print(f"DEBUG: Robot {self.robot_id} removing {len(self.my_reservations)} reservations.")
        for state in self.my_reservations:
            # Only delete if the reservation still belongs to this robot
            if Robot.reservation_table.get(state) is self:
                try:
                    del Robot.reservation_table[state]
                except KeyError:
                    # Should not happen, but handles potential race conditions
                    # print(f"WARN: Robot {self.robot_id} tried to remove non-existent reservation {state}", file=sys.stderr)
                    pass # Ignore if already deleted
        self.my_reservations.clear()

    def _reserve_path(self, path: Path) -> None:
        """
        Helper method to add reservations for a given path to the shared
        reservation table and the robot's local `my_reservations` set.
        Assumes the path is valid and overwrites any existing reservations
        (prioritized planning assumes higher priority robots plan first).
        """
        # print(f"DEBUG: Robot {self.robot_id} reserving path: {path}")
        for state in path:
            reserved_by = Robot.reservation_table.get(state)
            if reserved_by is not None and reserved_by is not self:
                 # This indicates a potential issue in planning order or conflict resolution
                 print(f"WARN: Robot {self.robot_id} overwriting reservation at {state} held by Robot {reserved_by.robot_id}", file=sys.stderr)

            Robot.reservation_table[state] = self
            self.my_reservations.add(state)

    def plan_path(self, grid: Grid, current_time: TimeStep) -> Optional[Path]:
        """
        Plans a collision-free path from the current position and time to the goal
        using time-expanded A* search. Treats robots at goals as permanent obstacles.

        Args:
            grid (Grid): The static environment grid (0=free, 1=obstacle).
            current_time (TimeStep): The simulation time from which planning starts.

        Returns:
            Optional[Path]: A list of (row, col, t) states from current_time to goal,
                            or None if no valid path is found within the deadline/limits.
        """
        rows, cols = grid.shape

        # --- Initial Checks ---
        if self.at_goal(): # If already at the goal when planning is requested
            stay_state: State = (self.x, self.y, current_time)
            # Ensure reservation exists for staying put at the current time
            if stay_state not in self.my_reservations:
                current_res = Robot.reservation_table.get(stay_state)
                if current_res is None: # If no one has it, reserve it
                    Robot.reservation_table[stay_state] = self
                    self.my_reservations.add(stay_state)
                elif current_res is not self: # Someone else has our goal cell reserved now? Problem!
                    print(f"ERROR: Robot {self.robot_id} at goal but Robot {current_res.robot_id} holds reservation {stay_state}", file=sys.stderr)
                    return None # Cannot stay at goal if someone else reserved it
            return [stay_state] # Path is just staying put at the current time

        # Clear previous reservations before planning a new path
        self.remove_reservations()

        # --- Create a planning grid that includes robots at goals as obstacles ---
        planning_grid = grid.copy()
        robots_at_goal = set()
        
        # Find all robots at goals and mark their positions as obstacles
        for state, robot in Robot.reservation_table.items():
            if robot is not self and robot.at_goal():
                planning_grid[robot.goal_x, robot.goal_y] = 1  # Mark as obstacle
                robots_at_goal.add((robot.goal_x, robot.goal_y))  # Save for debug/logging if needed
        
        # If debug info is wanted:
        # if robots_at_goal:
        #     print(f"DEBUG: Robot {self.robot_id} planning with {len(robots_at_goal)} robots at goals as obstacles")

        # --- A* Setup ---
        start_state: State = (self.x, self.y, current_time)
        # Priority queue stores: (f_score, g_score, state, parent_state)
        # f = g + h. Use g_score (time) as tie-breaker for states with same f_score.
        open_list: List[Tuple[float, int, State, Optional[State]]] = []
        initial_heuristic = self.heuristic(self.x, self.y)
        heapq.heappush(open_list, (initial_heuristic, 0, start_state, None)) # f = h, g = 0

        came_from: Dict[State, Optional[State]] = {start_state: None} # Stores path predecessors
        cost_so_far: Dict[State, int] = {start_state: 0} # g_scores (time steps)

        found_goal_state: Optional[State] = None

        # Set a reasonable planning horizon to prevent infinite loops in impossible scenarios
        # Max steps = manhattan distance + buffer, or grid size * 2, bounded by deadline if set.
        max_steps_from_start = self.manhattan_distance_to_goal() + max(rows, cols) # Generous buffer
        if self.deadline is not None:
            max_plan_time = self.deadline
        else:
            max_plan_time = current_time + max_steps_from_start # Avoid excessive planning without deadline

        # --- A* Search Loop ---
        while open_list:
            f_score, g_score, current_state, _ = heapq.heappop(open_list)
            x, y, t = current_state

            # --- Goal Check ---
            if x == self.goal_x and y == self.goal_y:
                # Found goal. Deadline check happens implicitly as nodes exceeding deadline aren't added/explored.
                found_goal_state = current_state
                break # Found the best path according to A*

            # --- Pruning ---
            # 1. Time Limit Pruning: Stop searching if time exceeds planning horizon or deadline
            if t >= max_plan_time:
                continue

            # 2. Already Found Better Path (Closed Set Check):
            # If we pulled a state from PQ but have already found a shorter path (lower g)
            # to it previously, skip processing this one.
            if g_score > cost_so_far.get(current_state, float('inf')):
                continue

            # --- Explore Neighbors ---
            for dx, dy in Robot.DIRECTIONS:
                next_x, next_y = x + dx, y + dy
                next_t = t + 1

                # --- Validity & Collision Checks for Neighbor ---
                # 1. Bounds Check
                if not (0 <= next_x < rows and 0 <= next_y < cols):
                    continue
                # 2. Static Obstacle Check (using planning_grid that includes robots at goals)
                if planning_grid[next_x, next_y] == 1:
                    continue
                # 3. Deadline Check (Check before adding neighbor)
                if self.deadline is not None and next_t > self.deadline:
                    continue

                neighbor_state: State = (next_x, next_y, next_t)

                # 4. Vertex Conflict Check (Check reservation table)
                reserved_by = Robot.reservation_table.get(neighbor_state)
                if reserved_by is not None and reserved_by is not self:
                    continue # Target cell reserved by another robot at target time

                # 5. Edge (Swap) Conflict Check
                # Check if another robot is currently at the neighbor cell (at time t)
                # AND plans to move into our current cell (at time t+1).
                swap_origin_state: State = (next_x, next_y, t) # Where the other robot would be
                swap_target_state: State = (x, y, next_t)      # Where we are, at the next timestep
                occupier_swap_origin = Robot.reservation_table.get(swap_origin_state)
                # Check if the potential swapper exists and is not us
                if occupier_swap_origin is not None and occupier_swap_origin is not self:
                    # Check if *that same robot* has reserved the cell we are currently in for the next timestep
                    if Robot.reservation_table.get(swap_target_state) is occupier_swap_origin:
                        continue # Swap conflict detected

                # --- Process Valid Neighbor ---
                new_g_score = g_score + 1 # Cost increases by 1 timestep
                # If this is the first time visiting neighbor OR we found a shorter path
                if new_g_score < cost_so_far.get(neighbor_state, float('inf')):
                    cost_so_far[neighbor_state] = new_g_score
                    priority = new_g_score + self.heuristic(next_x, next_y)
                    heapq.heappush(open_list, (priority, new_g_score, neighbor_state, current_state))
                    came_from[neighbor_state] = current_state

        # --- Path Reconstruction ---
        if found_goal_state is None:
            # No path found within limits/deadline
            # Set deadline_missed *only* if a deadline exists and might have been violated
            if self.deadline is not None and current_time < self.deadline:
                # If deadline exists but hasn't passed yet, failure implies blocked path
                self.deadline_missed = True # Or introduce a 'stuck' flag? For now, mark failed.
            # If deadline passed during planning, A* wouldn't have found it -> failed.
            elif self.deadline is not None and current_time >= self.deadline:
                self.deadline_missed = True
            # print(f"DEBUG: Robot {self.robot_id} failed to find path from ({self.x},{self.y}) at t={current_time}.")
            return None

        # Backtrack from the goal state to reconstruct the path
        path: Path = []
        current: Optional[State] = found_goal_state
        while current is not None:
            path.append(current)
            parent = came_from.get(current)
            # Cycle detection (optional sanity check - shouldn't happen in A*)
            # if parent in path:
            #     print(f"ERROR: Cycle detected in path reconstruction for Robot {self.robot_id}!", file=sys.stderr)
            #     return None # Indicate error
            current = parent
        path.reverse() # Reverse to get path from start to goal

        # Reserve the path in the shared table and local set
        self._reserve_path(path)
        return path
    
    def step(self, grid: Grid, all_robots: List['Robot'], current_time: TimeStep) -> None:
        """
        Executes one simulation time step for the robot.
        Handles validation, planning, dynamic conflict checks, and movement.
        Uses a replan-on-conflict strategy.

        Args:
            grid (Grid): The static environment grid.
            all_robots (List['Robot']): List of all robot instances (unused currently).
            current_time (TimeStep): The current simulation time step.
        """

        # --- 1. Initial Checks ---
        if current_time < self.start_time:
            # print(f"DEBUG: R{self.robot_id} not started (t={current_time} < start={self.start_time})")
            return # Not started yet

        if self.at_goal():
            # print(f"DEBUG: R{self.robot_id} already at goal.")
            # If at goal, ensure reservation for *waiting* at goal persists for next steps
            # Reserve multiple steps ahead to prevent other robots from planning through this position
            for t_offset in range(1, 6):  # Reserve 5 steps ahead
                goal_state_future: State = (self.goal_x, self.goal_y, current_time + t_offset)
                if goal_state_future not in self.my_reservations:
                    current_res = Robot.reservation_table.get(goal_state_future)
                    if current_res is None: # If no one has it, reserve it
                        Robot.reservation_table[goal_state_future] = self
                        self.my_reservations.add(goal_state_future)
            return # Already finished task

        if self.deadline_missed:
            # print(f"DEBUG: R{self.robot_id} already marked as failed.")
            return # Already failed

        # --- 2. Plan or Validate Path ---
        needs_plan = False
        if not self.path or self.path_index >= len(self.path):
            # Path is empty or we've finished executing the previous plan
            needs_plan = True
            # print(f"DEBUG: R{self.robot_id} needs plan (no/ended path) at t={current_time}")
        else:
            # Validate current state against the expected state in the path
            expected_state = self.path[self.path_index]
            if expected_state[0] != self.x or expected_state[1] != self.y or expected_state[2] != current_time:
                # Robot's current state doesn't match the plan's expectation for this time step
                needs_plan = True
                # print(f"DEBUG: R{self.robot_id} needs plan (state mismatch: plan={expected_state}, actual=({self.x},{self.y},{current_time}))")

        if needs_plan:
            # print(f"DEBUG: R{self.robot_id} planning at t={current_time} from ({self.x},{self.y})")
            new_path = self.plan_path(grid, current_time)
            if new_path is None:
                # Planning failed (no path found or hit deadline/limits)
                self.consecutive_planning_failures += 1
                # print(f"INFO: R{self.robot_id} failed planning at t={current_time}. Failure count: {self.consecutive_planning_failures}")
                
                # Mark as failed if: deadline exists and is violated, OR too many consecutive failures
                if (self.deadline is not None and current_time >= self.deadline) or \
                (self.consecutive_planning_failures >= self.max_planning_failures):
                    self.deadline_missed = True
                    print(f"INFO: R{self.robot_id} marked as failed after {self.consecutive_planning_failures} planning attempts at t={current_time}.")
                
                self.remove_reservations() # Clean up any potential remnants from failed plan
                self.path = [] # Ensure path is empty
                self.path_index = 0
                # Robot does not move this step
                return
            else:
                # Successfully planned a new path
                self.consecutive_planning_failures = 0  # Reset counter on successful planning
                self.path = new_path
                self.path_index = 0 # Start executing from the beginning of the new path
                # print(f"DEBUG: R{self.robot_id} new path planned (len={len(self.path)}): {self.path[:min(5, len(self.path))]}")


        # --- 3. Determine Next Intended State and Check Dynamic Conflicts ---
        # Ensure we have a valid path and index after potential planning/validation
        if not self.path or self.path_index >= len(self.path) or self.path[self.path_index][2] != current_time:
            # This implies planning failed or there's a logic error
            if self.path: # Log error only if path exists but is inconsistent
                print(f"CRITICAL ERROR: R{self.robot_id} invalid path state before move check! "
                    f"Path: {self.path}, Index: {self.path_index}, Time: {current_time}", file=sys.stderr)
                self.deadline_missed = True # Fail safe
                self.remove_reservations()
            # If path is empty (planning failed), normal exit, do nothing.
            return

        # Determine the state the robot INTENDS to be in at the END of this step (time t+1)
        # This is the state described by path[path_index + 1]
        intended_state_next_t: Optional[State] = None
        is_waiting = False
        if self.path_index + 1 < len(self.path):
            next_state_in_plan = self.path[self.path_index + 1]
            if next_state_in_plan[2] == current_time + 1:
                intended_state_next_t = next_state_in_plan
                is_waiting = (intended_state_next_t[0] == self.x and intended_state_next_t[1] == self.y)
            else:
                # This suggests a time gap in the plan, which shouldn't happen
                print(f"ERROR: R{self.robot_id} path time jump detected! Path={self.path}, Idx={self.path_index}, Time={current_time}", file=sys.stderr)
                # Assume wait as fallback? Risky. Invalidate plan.
                self.path = []
                self.path_index = 0
                self.remove_reservations()
                return
        else:
            # Reached end of path according to index. Should imply goal reached.
            # The intended action is to stay put (wait) at the goal.
            if not (self.x == self.goal_x and self.y == self.goal_y):
                print(f"WARN: R{self.robot_id} path ended at index {self.path_index} ({self.path[-1]}) but not at goal ({self.goal_x},{self.goal_y})", file=sys.stderr)
                # Path might have been truncated? Or planning error? Treat as stuck/wait.
            intended_state_next_t = (self.x, self.y, current_time + 1)
            is_waiting = True


        # Check for conflicts with the INTENDED state at t+1
        conflict_detected = False
        conflict_reason = ""
        reserved_by = Robot.reservation_table.get(intended_state_next_t)
        if reserved_by is not None and reserved_by is not self:
            conflict_detected = True
            conflict_reason = f"Vertex conflict with R{reserved_by.robot_id} at {intended_state_next_t}"

        # Check Swap conflict only if actually moving to a different cell
        if not conflict_detected and not is_waiting:
            # State other robot would be in to cause swap: (intended target pos, current time)
            swap_origin_state: State = (intended_state_next_t[0], intended_state_next_t[1], current_time)
            # State other robot would move into: (our current pos, next time)
            swap_target_state: State = (self.x, self.y, current_time + 1)
            occupier_swap_origin = Robot.reservation_table.get(swap_origin_state)
            # If the origin cell is occupied by someone else...
            if occupier_swap_origin is not None and occupier_swap_origin is not self:
                # ...and that same robot has reserved our current cell for the next step
                if Robot.reservation_table.get(swap_target_state) is occupier_swap_origin:
                    conflict_detected = True
                    conflict_reason = f"Swap conflict with R{occupier_swap_origin.robot_id} involving move to {intended_state_next_t[:2]}"

        # --- 4. Handle Conflict (Replan) or Execute Move ---
        executed_action = False # Flag to indicate if an action (move/wait) was determined for this step
        if conflict_detected:
            # print(f"INFO: R{self.robot_id} detected dynamic conflict ({conflict_reason}). Attempting replan at t={current_time}.")
            # Attempt a full replan. The conflicting reservation is now visible.
            new_path = self.plan_path(grid, current_time)

            if new_path is None:
                # Replan also failed. Robot is stuck this step.
                self.consecutive_planning_failures += 1
                # print(f"WARN: R{self.robot_id} failed to replan after conflict. Stuck at t={current_time}. Failures: {self.consecutive_planning_failures}")
                
                # Mark as failed if too many consecutive planning failures
                if self.consecutive_planning_failures >= self.max_planning_failures:
                    self.deadline_missed = True
                    print(f"INFO: R{self.robot_id} marked as failed after {self.consecutive_planning_failures} replanning attempts at t={current_time}.")
                    
                self.path = [] # Invalidate path, force replan next time
                self.path_index = 0
                # Do not move, action not executed
            else:
                # Replan succeeded, use the new path
                self.consecutive_planning_failures = 0  # Reset counter on successful replanning
                # print(f"DEBUG: R{self.robot_id} replanned successfully after conflict. New path length: {len(new_path)}")
                self.path = new_path
                self.path_index = 0
                # Determine the action based on the *new* path's first step
                if self.path_index + 1 < len(self.path):
                    state_after_move = self.path[self.path_index + 1]
                    if state_after_move[2] == current_time + 1:
                        # Execute move from new plan
                        self.x = state_after_move[0]
                        self.y = state_after_move[1]
                        executed_action = True
                    else: # New plan starts with wait or has time jump (error case handled above)
                        # Position doesn't change if waiting
                        executed_action = True # "Wait" is the executed action
                else: # New path is just current state (e.g., plan resulted in staying put)
                    executed_action = True # "Wait" is the action

        else:
            # No conflict detected in reservations, but we need to check for robots physically at the intended position
            target_x, target_y = intended_state_next_t[0], intended_state_next_t[1]
            
            # Check if any other robot is physically at our intended next position
            occupied_by_robot_at_goal = False
            for other_robot in all_robots:
                # Skip ourselves and robots not yet started
                if other_robot is self or current_time < other_robot.start_time:
                    continue
                    
                # Check if other robot is at the target position AND at its goal
                if (other_robot.x == target_x and other_robot.y == target_y and 
                    other_robot.at_goal()):
                    occupied_by_robot_at_goal = True
                    conflict_reason = f"Runtime collision with R{other_robot.robot_id} at goal position ({target_x},{target_y})"
                    # print(f"INFO: R{self.robot_id} detected physical robot {other_robot.robot_id} at target position. Replanning.")
                    break
                    
            if occupied_by_robot_at_goal:
                # Similar to conflict handling - try to replan
                new_path = self.plan_path(grid, current_time)
                
                if new_path is None:
                    # Replan failed, increment failure counter and potentially mark as failed
                    self.consecutive_planning_failures += 1
                    if self.consecutive_planning_failures >= self.max_planning_failures:
                        self.deadline_missed = True
                        print(f"INFO: R{self.robot_id} marked as failed after {self.consecutive_planning_failures} replanning attempts at t={current_time}.")
                        
                    self.path = [] # Invalidate path, force replan next time
                    self.path_index = 0
                    # Do not move, action not executed
                else:
                    # Replan succeeded, use the new path
                    self.consecutive_planning_failures = 0
                    self.path = new_path
                    self.path_index = 0
                    if self.path_index + 1 < len(self.path):
                        state_after_move = self.path[self.path_index + 1]
                        if state_after_move[2] == current_time + 1:
                            # Execute move from new plan
                            self.x = state_after_move[0]
                            self.y = state_after_move[1]
                            executed_action = True
                        else:
                            # Position doesn't change if waiting
                            executed_action = True # "Wait" is the executed action
                    else:
                        executed_action = True # "Wait" is the action
            else:
                # No robots at the target position, execute move normally
                self.x = intended_state_next_t[0]
                self.y = intended_state_next_t[1]
                executed_action = True
        

        # --- 5. Advance Path Index if an Action Was Taken ---
        if executed_action:
            # We processed the state for current_time, advance index for the next step
            self.path_index += 1
            # Reset consecutive failures counter since we executed an action
            self.consecutive_planning_failures = 0
            self.total_moves += 1
        # else: Robot got stuck, index remains same, will likely replan next step


        # --- 6. Final Goal Check ---
        # Check if the *new* position (after move/wait) is the goal
        if self.x == self.goal_x and self.y == self.goal_y:
            if not self.at_goal_flag:
                # print(f"INFO: R{self.robot_id} reached goal ({self.x},{self.y}) at end of step {current_time}.")
                self.at_goal_flag = True
                # Optional: Trim path if desired, reduces memory for reservations
                # self.path = self.path[:self.path_index]
                self.arrival_time = current_time + 1   # reached at the END of this step
                # Reserve goal position for several steps ahead to ensure other robots treat it as obstacle
                for t_offset in range(1, 20):  # Reserve 20 steps ahead
                    goal_state_future: State = (self.goal_x, self.goal_y, current_time + t_offset)
                    if goal_state_future not in self.my_reservations:
                        current_res = Robot.reservation_table.get(goal_state_future)
                        if current_res is None:
                            Robot.reservation_table[goal_state_future] = self
                            self.my_reservations.add(goal_state_future)