from __future__ import print_function
import gym
import Env1Trainer
import math

env = gym.make('Env-1-v0')
env.reset()
solved = False
accuracy_gained = 0.0
final_observation = None
for i_episode in range(2):
    observation = env.reset()
    action = env.action_space.sample()
    print('Action: ' + str(action))
    last_reward = 0.0
    for t in range(20):
        print('i_episode:', i_episode, 't:', t, observation)
        observation, reward, done = env.step(action)
        print('--------------------------')
        print('Action: ' + str(action))
        if last_reward - reward > 0.01:
            action = env.action_space.sample()
            observation = env.recover_last_parameters()
            print('IT WAS A BAD ACTION!!!!')
        else:
            if math.fabs(reward - last_reward) < 0.05:
                if action is 1:
                    action = 3
                else:
                    action = env.action_space.sample()
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
