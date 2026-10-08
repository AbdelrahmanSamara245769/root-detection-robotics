import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from ot2_gym_wrapper import OT2ReachEnv
from project_paths import MODELS_DIR

def main():
    model_path = os.path.join(MODELS_DIR, "ppo_ot2_final.zip")
    stats_path = os.path.join(MODELS_DIR, "vec_normalize.pkl")

    if not os.path.exists(stats_path):
        print(f"Error: {stats_path} not found. Did you finish training?")
        return

    # 1. Recreate the environment exactly as it was during training
    env = DummyVecEnv([lambda: OT2ReachEnv(render=True)])
    
    # 2. Load the normalization stats (This is what was missing!)
    env = VecNormalize.load(stats_path, env)
    env.training = False # DO NOT update mean/std during test
    env.norm_reward = False # DO NOT normalize rewards during test

    # 3. Load the model
    model = PPO.load(model_path, env=env)

    successes = 0
    total_episodes = 20
    errors = []

    for ep in range(total_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            if done:
                final_dist = info[0]['dist']
                errors.append(final_dist)
                if final_dist < 0.001:
                    successes += 1
                print(f"Ep {ep+1}: Final Error = {final_dist:.4f}m")

    print("\n=== FINAL EVALUATION ===")
    print(f"Success Rate: {successes/total_episodes:.2%}")
    print(f"Mean Error: {np.mean(errors):.5f}m")

if __name__ == "__main__":
    main()