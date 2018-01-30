from gym.envs.registration import register

register(
    id='Env-1-v0',
    entry_point='Env1Trainer.env.env1:Env1Class',
)