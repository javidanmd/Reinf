from __future__ import print_function
import gym
import Env1Trainer

env = gym.make('Env-1-v0')
env.reset()
solved = False
accuracy_gained = 0.0
final_observation = None
for i_episode in range(2):
    observation = env.reset()
    action = env.action_space.sample()
    last_reward = 0.0
    for t in range(20):
        print('i_episode:', i_episode, 't:', t, observation)
        observation, reward, done = env.step(action)
        if last_reward - reward > 0.01:
            action = env.action_space.sample()
            env.recover_last_parameters()
            print('IT WAS A BAD ACTION!!!!')
        else:
            last_reward = reward
        print(reward)
        if reward >= 0.980:
            solved = True
            accuracy_gained = reward
            final_observation = observation
            break
        elif done:
            print('Exceeded Tests!')
    if solved:
        print('Solved!!!!!!!')
        break

print('Best accuracy is:', accuracy_gained)
print(final_observation)
