import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=10, spawn_border=3):
        self.size = size
        self.spawn_border = spawn_border
        self.step_dir = [[-1,0],[1,0],[0,1],[0,-1]]
        head = np.random.randint(spawn_border,size-spawn_border, size=2)
        body_dir = self.step_dir[np.random.randint(4)]
        self.snake = [head, head+body_dir]
        self.food = np.random.randint(0, size, size=2)
        #Prevent head from spawning in food.
        while (self.food == head).all():
            self.food = np.random.randint(0, size, size=2)
        self.frame = np.zeros((size,size))
        
    def update_frame(self):
        self.frame = np.zeros((self.size,self.size))
        for piece in self.snake:
            self.frame[piece[0], piece[1]] = 255
        self.frame[self.food[0], self.food[1]] = 125

    def step(self, action):
        reward = 0
        done = False
        info = {None}
        #Check if movement is in direction of body. If not, take that action.
        if (self.snake[0] + self.step_dir[action] == self.snake[1]).all():
            move = self.snake[1] - self.snake[0]
            #TODO: Fix this hack.
            if (self.snake[0] + move == self.snake[1]).all():
                move *= -1
            self.snake.insert(0, self.snake[0] + move)
        else:
            self.snake.insert(0, self.snake[0] + self.step_dir[action])
        #Check if head is in same position as food.
        if (self.snake[0] != self.food).any():
            self.snake.pop()
        else:
            self.food = np.random.randint(0, self.size, size=2)
            reward = 1
        #Check if head is out of bounds or overlapped with body.
        head_body = False
        for piece in self.snake[1:]:
            if (self.snake[0] == piece).all():
                head_body = True
        border = ((self.snake[0] - [self.size, self.size]) == 0).any()
        border2 = (self.snake[0] < 0).any()
        if border or border2 or head_body:
            self.reset()
            reward = -1
            done = True
        #re-draw the frame.
        self.update_frame()
        return self.frame, reward, done, info

    def reset(self):
        self.__init__(self.size, self.spawn_border)
        return self.frame

    def render(self, mode='human'):
        return self.frame

    def close(self):
        self.__init__(self.size, self.spawn_border)