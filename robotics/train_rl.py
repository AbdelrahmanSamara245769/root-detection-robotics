import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from ot2_gym_wrapper import OT2ReachEnv
from project_paths import MODELS_DIR, OUTPUT_DIR

# 1. Create the ONE and ONLY Environment
# We remove the eval_env to prevent PyBullet conflicts
env = OT2ReachEnv(render=False, max_steps=1000)
env = DummyVecEnv([lambda: env])

# 2. Normalize
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# 3. Define PPO Model (CPU is fine, but if you have CUDA installed, use it)
# We use a smaller n_steps to update more frequently, which helps early convergence
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    batch_size=64,
    n_steps=2048,
    device="auto",  # Will pick CUDA if fixed, CPU if not. Both are fine now.
    tensorboard_log=os.path.join(OUTPUT_DIR, "logs")
)

# 4. Train
# Since we removed the eval callback, we just train for a fixed duration.
TIMESTEPS = 300000 
print(f"Training for {TIMESTEPS} timesteps...")
start_time = time.time()

model.learn(total_timesteps=TIMESTEPS, progress_bar=True)

end_time = time.time()
print(f"Training complete. Duration: {(end_time - start_time)/60:.2f} minutes")

# 5. Save EVERYTHING
os.makedirs(MODELS_DIR, exist_ok=True)
model.save(os.path.join(MODELS_DIR, "ppo_ot2_final"))
env.save(os.path.join(MODELS_DIR, "vec_normalize.pkl")) # Critical!
print("Model and Normalization stats saved.")