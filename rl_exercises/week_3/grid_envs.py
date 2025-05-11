from time import sleep

import gymnasium as gym
from rl_exercises.week_3.sarsa_lambda import EpsilonGreedyPolicy, SARSALambdaAgent


def episode(env, agent, evaluate=False):
    prev_state, info = env.reset(seed=46)
    prev_state = (prev_state["direction"], str(prev_state["image"]))
    state = prev_state

    episode_over = False
    while not episode_over:
        action = agent.predict_action(state, evaluate)
        state, reward, terminated, truncated, info = env.step(action)
        state = (state["direction"], str(state["image"]))
        episode_over = terminated or truncated

        if reward != 0.0:
            print("SUCCESS")

        agent.update_agent(
            prev_state, action, reward, state, agent.predict_action(state), episode_over
        )
        prev_state = state


env = gym.make("MiniGrid-FourRooms-v0", render_mode="none")
policy = EpsilonGreedyPolicy(env, epsilon=0.8, seed=43)
agent = SARSALambdaAgent(env, policy, alpha=0.5, gamma=0.9, lam=0.5)

for i in range(10):
    print(f"Episode {i}")
    episode(env, agent)

env = gym.make("MiniGrid-FourRooms-v0", render_mode="human")
print("Evaluation!")
episode(env, agent, True)
sleep(5)

env.close()
