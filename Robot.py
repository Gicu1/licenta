import random

class Robot:
    def __init__(self, start_x, start_y, goal_x, goal_y):
        self.x = start_x
        self.y = start_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.path = [(self.x, self.y)]  # Lista cu traiectoria parcursă
        self.speed = 1  # Viteza de deplasare 

    def move_towards_goal(self):
        # Calcularea direcției către destinație
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        distance = (dx**2 + dy**2)**0.5
        if distance > 0:
            # Actualizarea poziției pe direcția goal
            self.x += self.speed * (dx / distance)
            self.y += self.speed * (dy / distance)
            # Adăugarea noii poziții în traiectorie
            self.path.append((self.x, self.y))
