import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=10, spawn_border=3):
        self.size = size
        self.spawn_border = spawn_border
        self.num_envs = 1
        self.step_dir = [[-1,0],[1,0],[0,1],[0,-1]]
        self.l = 0
        self.ep = 0
        self.R = 0

        self.seed()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.size, self.size, 3), dtype=np.uint8)
        self.reset_game()

    def reset_game(self):
        head = np.random.randint(self.spawn_border,self.size-self.spawn_border, size=2)
        body_dir = self.step_dir[np.random.randint(4)]
        self.snake = [head, head+body_dir]
        self.food = np.random.randint(0, self.size, size=2)
        #Prevent head from spawning in food.
        while (self.food == head).all():
            self.food = np.random.randint(0, self.size, size=2)
        self.frame = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def update_frame(self):
        self.frame = np.zeros((self.size,self.size, 3), dtype=np.uint8)
        self.frame[self.food[0], self.food[1], 1] = 255
        for idx, piece in enumerate(self.snake):
            if idx == 0:
                self.frame[piece[0], piece[1], 2] = 255
            else:
                self.frame[piece[0], piece[1], 0] = 255

    def step(self, action):
        self.l += 1
        reward = 0
        done = False
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
            self.R += 1
        #Check if head is out of bounds or overlapped with body.
        head_body = False
        for piece in self.snake[1:]:
            if (self.snake[0] == piece).all():
                head_body = True
        border = ((self.snake[0] - [self.size, self.size]) == 0).any()
        border2 = (self.snake[0] < 0).any()
        if border or border2 or head_body or self.l >= 1000:
            self.reset()
            done = True
        #re-draw the frame after action.
        self.update_frame()
        info = {'episode': {'episode': self.ep, 'r': self.R, 'l':self.l}}
        if done:
            self.R = 0
            self.l = 0
            self.ep += 1
        return self.frame, reward, done, info
    
    def step_async(self, actions):
        self.obs, self.reward, self.done, self.info = self.step(actions[0])
   
    def step_wait(self):
        return self.obs, np.array([self.reward]), np.array([self.done]), np.array([self.info])

    def reset(self):
        self.reset_game()
        return self.frame

    def render(self, mode='human'):
        return self.frame

    def close(self):
        self.__init__(self.size, self.spawn_border)