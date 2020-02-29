import gym
import Snake
import tensorflow as tf
from baselines.ppo2 import ppo2
import os.path as osp
from baselines.common.models import build_impala_cnn
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger

LOG_DIR = './snake_train_folder'

learning_rate = 5e-4
ent_coef = .01
gamma = .999
lam = .95
nsteps = 256
nminibatches = 8
ppo_epochs = 3
clip_range = .2
use_vf_clipping = True

format_strs = ['csv', 'stdout']
logger.configure(dir=LOG_DIR, format_strs=format_strs)

logger.info("creating environment")
venv = gym.make('snake-v0')

venv = VecMonitor(
    venv=venv, filename=None, keep_buf=100,
)

logger.info("creating tf session")
config = tf.ConfigProto()
sess = tf.Session(config=config)
sess.__enter__()

conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

logger.info("training")
final_model = ppo2.learn(
    env=venv,
    network=conv_fn,
    total_timesteps=int(1e4),
    save_interval=0,
    nsteps=nsteps,
    nminibatches=nminibatches,
    lam=lam,
    gamma=gamma,
    noptepochs=ppo_epochs,
    log_interval=1,
    ent_coef=ent_coef,
    mpi_rank_weight=0,
    clip_vf=use_vf_clipping,
    lr=learning_rate,
    cliprange=clip_range,
    update_fn=None,
    init_fn=None,
    vf_coef=0.5,
    max_grad_norm=0.5,
)

savepath = osp.join(logger.get_dir(), 'final')
final_model.save(savepath)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

obs = venv.reset()
frames = []
fig = plt.figure()

for t in range(500):
    actions, values, states, neglogpacs = final_model.step(obs)
    obs[:], rewards, dones, infos = venv.step(actions)
    im = plt.imshow(obs.astype(np.uint8), animated=True)
    frames.append([im])

ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                repeat_delay=1000)

ani.save(osp.join(logger.get_dir(), 'dynamic_images.mp4'))

plt.show()
