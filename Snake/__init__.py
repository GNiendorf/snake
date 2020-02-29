from gym.envs.registration import register

register(
    id='Maze',
    entry_point='Maze.envs:MazeEnv',
)