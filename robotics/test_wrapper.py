import numpy as np
from ot2_gym_wrapper import OT2ReachEnv

def main():
    # 1. Initialize environment
    # Setting render=True allows you to see the robot move randomly
    env = OT2ReachEnv(render=True, action_scale=0.1)

    print("--- OT-2 Wrapper Sanity Test ---")
    
    # 2. Reset the environment
    obs, info = env.reset()
    
    # Check observation shape - must be (6,) for our new setup
    print(f"Observation shape: {obs.shape}") 
    print(f"Initial Pipette Position: {obs[:3]}")
    print(f"Initial Target Position:  {obs[3:6]}")

    # 3. Random action loop
    for step in range(1, 201):
        # Sample a random normalized action [-1, 1]
        action = env.action_space.sample()
        
        # Step the simulation
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print stats every 50 steps
        if step % 50 == 0:
            # Note: We now pull 'dist' from the info dict
            current_dist = info.get('dist', 0.0)
            print(f"Step: {step:3d} | Dist: {current_dist:.4f}m | Reward: {reward:.4f}")

        if terminated:
            print(f"Target reached at step {step}!")
            break
            
        if truncated:
            print("Episode reached max steps.")
            break

    env.close()
    print("--- Test Complete: Wrapper is ready for training! ---")

if __name__ == "__main__":
    main()