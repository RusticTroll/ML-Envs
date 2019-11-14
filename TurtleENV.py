import gym
import gym.spaces as spaces
from random import randint
import numpy as np
from math import sqrt
import turtle
import time

turtle.tracer(0)
screen = turtle.Screen()
screen.screensize(1000, 1000)


class turtleTrack(gym.Env):

    def __init__(self):
        self.crash = turtle.Turtle()
        self.action_space = spaces.Discrete(31)
        self.observation_space = spaces.Box(np.array(1), np.array(1), dtype=np.float32)

        self.crash.pu()
        self.crash.goto(0, -375)
        self.crash.pd()
        self.crash.begin_fill()
        self.crash.circle(375)
        self.crash.end_fill()
        screen.update()
        self.crash.pu()
        
        self.reset()
        self.usedGates = []
        self.renderMove = False

    def reset(self):
        self.crash.setpos(-1, -450)
        self.reward = 0
        self.usedGates = []
        self.positions = [(-1, -450)]
        return self.crash.pos()

    def distance(self, point1, point2):
        if not isinstance(point1, (list, tuple)) or not isinstance(point2, (list, tuple)):
            raise TypeError("This function only accepts lists or tuples")
        return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def step(self, action):
        oldPos = self.crash.pos()
        self.crash.setheading(self.crash.heading() - (action - 15))
        self.crash.forward(50)
        x = self.crash.xcor()
        y = self.crash.ycor()
        self.positions.append(self.crash.pos())
        if self.renderMove:
            screen.update()
        if self.distance(self.crash.pos(), (0, 0)) <= 375:
            return self.crash.pos(), -1, True, {}
        elif x < -500 or x > 500 or y < -500 or y > 500:
            return self.crash.pos(), self.reward - 1, True, {}
        elif self.rewardGate(oldPos):
            self.reward += 1
            print("Reward:" + str(self.reward))
            print(self.positions)
            time.sleep(1/60)
        return self.crash.pos(), self.reward, False, {}

    def rewardGate(self, oldPos):
        if oldPos[1] <= 0 < self.crash.pos()[1] and not self.usedGates.__contains__(0) and self.crash.xcor() < 0:
            self.usedGates.append(0)
            return True
        elif oldPos[1] >= 0 > self.crash.pos()[1] and not self.usedGates.__contains__(1) and self.crash.xcor() > 0:
            self.usedGates.append(1)
            return True
        elif oldPos[0] <= 0 < self.crash.pos()[0] and not self.usedGates.__contains__(2) and self.crash.ycor() > 0:
            self.usedGates.append(2)
            return True
        elif oldPos[0] >= 0 > self.crash.pos()[0] and not self.usedGates.__contains__(3) and self.crash.ycor() < 0:
            if len(self.usedGates) == 3:
                self.usedGates.clear()
            self.usedGates.append(3)
            return True
        else: return False

    def render(self, mode):
        if mode == "human":
            screen.update()

print("The turtle track has been successfully initialized")