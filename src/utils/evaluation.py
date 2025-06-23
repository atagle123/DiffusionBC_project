import numpy as np
import gym
from d4rl import get_normalized_score

def evaluate_parallel(
    policy_fn, envs, env_entry: str, num_episodes: int, seed: int=42, max_num_steps: int=1000,
) -> dict[str, float]:

    observations = np.array([env.reset() for env in envs])
    dones = np.zeros(num_episodes, dtype=bool)
    episode_returns = np.zeros(num_episodes)

    # Iterate over environment steps
    while not np.all(dones): # TODO add a max number of steps
        actions = policy_fn(observations)

        # Collect rewards and update states
        next_observations = []
        rewards = []
        next_dones = []

        for i, (env, done) in enumerate(zip(envs, dones)):
            observation = observations[i]
            if not done:
                action = actions[i]
                observation, reward, done, _ = env.step(action)
                next_observations.append(observation)
                rewards.append(reward)
                next_dones.append(done)

            else:
                # If the episode is done, we set the reward to 0 and continue with the final state
                next_observations.append(observation)
                rewards.append(0.0)
                next_dones.append(True)

        # Update the states for each environment
        observations = np.array(next_observations)
        dones = np.array(next_dones)
        episode_returns += np.array(rewards)

    scores = get_normalized_score(env_name=env_entry, score=episode_returns) * 100
    scores_mean = np.mean(scores)
    scores_std = np.std(scores)

    return {"mean":scores_mean, "std":scores_std}