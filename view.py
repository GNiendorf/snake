import os.path as osp
import os

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from baselines import logger
from baselines.ppo2 import ppo2
from baselines.common.models import build_impala_cnn

import Snake

LOG_DIR = './snake_train_folder'
format_strs = ['csv', 'stdout']
logger.configure(dir=LOG_DIR, format_strs=format_strs)
#Goes much faster without gpu for some reason...
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

venv = gym.make('snake-v0')

config = tf.ConfigProto()
sess = tf.Session(config=config)
sess.__enter__()

conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

final_model = ppo2.learn(
    env=venv,
    network=conv_fn,
    total_timesteps=0,
    mpi_rank_weight=0,
    update_fn=None,
    init_fn=None,
)

loadpath = osp.join(logger.get_dir(), 'final')
final_model.load(loadpath)

obs = venv.reset()
fig = plt.figure()
frames = []
done = 0
eps = 5

while done < eps:
    actions, values, states, neglogpacs = final_model.step(obs)
    obs[:], rewards, dones, infos = venv.step(actions)
    done += dones[0]
    im = plt.imshow(obs.astype(np.uint8), animated=True)
    frames.append([im])

ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                repeat_delay=1000)

ani.save(osp.join(logger.get_dir(), 'snake.mp4'))

plt.show()
