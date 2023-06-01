import sys
import gym
import numpy as np
import pickle
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from itertools import count

def save_expert_traj(env_name, env, model, nr_trajectories=10):
    num_steps = 0
    expert_traj = {'obs':[], 'acs':[], 'r':[], 'd':[]}

    if isinstance(nr_trajectories, list):
        nr_trajectories = nr_trajectories[0]

    for i_episode in count():
        ob = env.reset()
        done = False
        total_reward = 0

        while not done:
            ac, _states = model.predict(ob)
            next_ob, reward, done, _ = env.step(ac)
            ob = next_ob
            total_reward += reward
            expert_traj['obs'].append(ob)
            expert_traj['acs'].append(ac)
            expert_traj['r'].append(reward)
            expert_traj['d'].append(done)
            num_steps += 1

        print("episode:", i_episode, "reward:", total_reward)

        if i_episode > nr_trajectories:
            break

    filename = f"demo_hf_{env_name}_{nr_trajectories}.pkl" 
    # expert_traj = np.stack(expert_traj)
    print(len(expert_traj['obs']))
    pickle.dump(expert_traj, open(filename,'wb'))
    # np.save(filename, expert_traj)

if __name__ == '__main__':
    env_name = sys.argv[1]
    algo = sys.argv[2]
    nr_traj = int(sys.argv[3])

    checkpoint = load_from_hub(
        repo_id=f"sb3/{algo}-{env_name}",
        filename=f"{algo}-{env_name}.zip",
    )
    custom_objects = {
      "learning_rate": 0.0,
      "lr_schedule": lambda _: 0.0,
      "clip_range": lambda _: 0.0,
    }
    if algo == 'sac':
        model = SAC.load(checkpoint, custom_objects=custom_objects)
    else:
        model = PPO.load(checkpoint, custom_objects=custom_objects)
    env = make_atari_env(env_name, n_envs=1, seed=0)
    # Stack 4 frames
    env = VecFrameStack(env, n_stack=4)
    # env = gym.make(env_name)
    # env = gym.wrappers.ResizeObservation(env, (84, 84))
    # env = gym.wrappers.GrayScaleObservation(env)
    # env = gym.wrappers.FrameStack(env, 4)
    # env = AtariWrapper(env) 

    save_expert_traj(env_name, env, model, nr_traj)


    

    

