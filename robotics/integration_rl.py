"""
Task 12 — System Integration (RL Controller)
Integrates the PPO RL controller with the CV pipeline to autonomously
inoculate root tips detected in specimen plates.

Uses the same coordinate transformation as the PID integration,
but swaps the controller for a trained PPO agent from Stable Baselines 3.
"""

import os
import json
import time
import numpy as np
import imageio

# ──────────────────────────────────────────────
# WORKING DIRECTORY SETUP
# ──────────────────────────────────────────────
from project_paths import CV_DATA_PATH, MODELS_DIR, OUTPUT_DIR, enter_simulation_dir

enter_simulation_dir()

# ──────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────
from sim_class import Simulation
from stable_baselines3 import PPO
import pybullet as p

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
MAX_STEPS_PER_TARGET = 1000
CONVERGENCE_TOL = 0.001       # 1mm
HOLD_STEPS = 10
ACTION_SCALE = 0.08           # must match training config

MODEL_PATH = os.path.join(MODELS_DIR, "ppo_ot2_final.zip")
GIF_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "integration")
os.makedirs(GIF_OUTPUT_DIR, exist_ok=True)

PLATE_TO_KEY = {
    "test_image_10.png": "test_image_10",
    "test_image_14.png": "test_image_14",
    "test_image_16.png": "test_image_16",
}

GIF_FPS = 20
FRAME_CAPTURE_INTERVAL = 5
GIF_WIDTH = 640
GIF_HEIGHT = 480


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────

def get_pipette_position(state):
    """Extract the pipette position from the simulation state."""
    key = list(state.keys())[0]
    return np.array(state[key]["pipette_position"], dtype=np.float32)


def load_root_tips(json_path):
    """Load pre-computed root tip coordinates from the CV pipeline."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def identify_plate(sim, plate_to_key):
    """Identify which plate the simulation loaded."""
    plate_path = sim.get_plate_image()
    plate_filename = os.path.basename(plate_path)
    print(f"Simulation loaded plate: {plate_filename}")
    if plate_filename in plate_to_key:
        return plate_to_key[plate_filename]
    else:
        print(f"WARNING: Unrecognized plate '{plate_filename}'")
        return None


def capture_frame(frames):
    """Capture a single high-quality frame from the simulation camera."""
    camera_pos = [1, 0, 1]
    camera_target = [-0.3, 0, 0]
    up_vector = [0, 0, 1]
    fov = 50
    aspect = GIF_WIDTH / GIF_HEIGHT

    width, height, rgbImg, _, _ = p.getCameraImage(
        width=GIF_WIDTH, height=GIF_HEIGHT,
        viewMatrix=p.computeViewMatrix(camera_pos, camera_target, up_vector),
        projectionMatrix=p.computeProjectionMatrixFOV(fov, aspect, 0.1, 100.0)
    )
    frame = np.array(rgbImg, dtype=np.uint8).reshape(GIF_HEIGHT, GIF_WIDTH, 4)
    frames.append(frame[:, :, :3])


def build_observation(pipette_pos, target):
    """
    Build the 6D observation vector matching the gym wrapper format:
    [pipette_x, pipette_y, pipette_z, target_x, target_y, target_z]
    """
    return np.concatenate([pipette_pos, target]).astype(np.float32)


def move_to_target_rl(sim, model, target, frames):
    """
    Use the RL agent to move the pipette to the target position.

    Returns:
        success (bool), steps_taken (int), final_error (float)
    """
    success_counter = 0

    for step in range(1, MAX_STEPS_PER_TARGET + 1):
        # Get current state
        state = sim.run([[0, 0, 0, 0]])
        pos = get_pipette_position(state)

        # Build observation (same format as training)
        obs = build_observation(pos, target)

        # Get action from trained agent
        action, _ = model.predict(obs, deterministic=True)

        # Clip and scale action (matching training config)
        action = np.clip(action, -1.0, 1.0)
        vel = action * ACTION_SCALE

        # Apply action with 10 physics steps (matching training config)
        sim.run([[vel[0], vel[1], vel[2], 0]], num_steps=10)

        # Compute error
        error = target - pos
        error_norm = float(np.linalg.norm(error))

        # Capture frame periodically
        if step % FRAME_CAPTURE_INTERVAL == 0:
            capture_frame(frames)

        # Progress logging
        if step % 200 == 0:
            print(f"    Step {step} | Error: {error_norm:.4f}m")

        # Convergence check with settling
        if error_norm < CONVERGENCE_TOL:
            success_counter += 1
            if success_counter >= HOLD_STEPS:
                print(f"    Converged at step {step} | Error: {error_norm:.4f}m")
                return True, step, error_norm
        else:
            success_counter = 0

    # Did not converge
    final_state = sim.run([[0, 0, 0, 0]])
    final_pos = get_pipette_position(final_state)
    final_error = float(np.linalg.norm(target - final_pos))
    print(f"    WARNING: Did not converge after {MAX_STEPS_PER_TARGET} steps | Error: {final_error:.4f}m")
    return False, MAX_STEPS_PER_TARGET, final_error


def drop_inoculum(sim, frames):
    """Drop inoculum at the current pipette position and capture frames."""
    print("    Dropping inoculum...")
    sim.run([[0, 0, 0, 1]])
    capture_frame(frames)
    for _ in range(30):
        sim.run([[0, 0, 0, 0]])
        capture_frame(frames)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run_integration_rl():
    """Run the autonomous inoculation pipeline using the RL controller."""

    print("=" * 60)
    print("TASK 12 — AUTONOMOUS ROOT TIP INOCULATION (RL)")
    print("=" * 60)

    # 1. Load root tip data
    root_tip_data = load_root_tips(CV_DATA_PATH)
    print(f"Loaded root tip data for {len(root_tip_data)} images")

    # 2. Load RL model
    print(f"\nLoading RL model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    print("RL model loaded successfully")

    # 3. Start simulation
    print("\nInitializing simulation...")
    sim = Simulation(num_agents=1, render=False, rgb_array=False)

    # 4. Identify plate
    plate_key = identify_plate(sim, PLATE_TO_KEY)

    if plate_key is None:
        print("ERROR: Could not identify plate. Closing.")
        sim.close()
        return

    if plate_key not in root_tip_data:
        print(f"ERROR: No root tip data for '{plate_key}'. Closing.")
        sim.close()
        return

    # Check if we already have a GIF for this plate
    gif_path = os.path.join(GIF_OUTPUT_DIR, f"integration_rl_{plate_key}.gif")
    if os.path.exists(gif_path):
        print(f"RL GIF already exists for {plate_key}, skipping. Delete it to re-run.")
        sim.close()
        return

    # 5. Get targets
    tips = root_tip_data[plate_key]
    valid_tips = [(i, t) for i, t in enumerate(tips) if t is not None]
    print(f"\nPlate: {plate_key}")
    print(f"Root tips found: {len(valid_tips)}/5")

    if len(valid_tips) == 0:
        print("ERROR: No valid root tips. Closing.")
        sim.close()
        return

    # 6. Inoculate using RL
    frames = []
    results_log = []
    capture_frame(frames)

    print("\n" + "=" * 60)
    print("STARTING AUTONOMOUS INOCULATION (RL CONTROLLER)")
    print("=" * 60)

    total_start_time = time.time()

    for lane, tip_data in valid_tips:
        target = np.array(tip_data["robot_coords_m"], dtype=np.float32)
        pixel = tip_data["pixel_original"]
        mm = tip_data["mm_plate_relative"]

        print(f"\n--- Plant {lane} ---")
        print(f"  Pixel (y,x): ({pixel[0]}, {pixel[1]})")
        print(f"  Plate mm (y,x): ({mm[0]:.1f}, {mm[1]:.1f})")
        print(f"  Robot target (x,y,z): ({target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f})")

        success, steps, final_error = move_to_target_rl(
            sim, model, target, frames
        )

        drop_inoculum(sim, frames)

        results_log.append({
            "lane": lane,
            "target": target.tolist(),
            "success": success,
            "steps": steps,
            "final_error_m": final_error,
            "final_error_mm": final_error * 1000
        })

        print(f"  Result: {'SUCCESS' if success else 'PARTIAL'} | Error: {final_error*1000:.2f}mm")

    total_time = time.time() - total_start_time

    # 7. Summary
    print("\n" + "=" * 60)
    print("INOCULATION COMPLETE (RL)")
    print("=" * 60)
    print(f"Plate: {plate_key}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Plants inoculated: {len(valid_tips)}/5")

    for r in results_log:
        status = "OK" if r["success"] else "WARN"
        print(f"  Lane {r['lane']}: [{status}] error={r['final_error_mm']:.2f}mm, steps={r['steps']}")

    avg_error = np.mean([r["final_error_m"] for r in results_log])
    max_error = np.max([r["final_error_m"] for r in results_log])
    successes = sum(1 for r in results_log if r["success"])

    print(f"\nAvg error: {avg_error*1000:.2f}mm")
    print(f"Max error: {max_error*1000:.2f}mm")
    print(f"Converged: {successes}/{len(results_log)}")

    # 8. Save GIF
    if frames:
        imageio.mimsave(gif_path, frames, fps=GIF_FPS)
        print(f"\nGIF saved: {gif_path} ({len(frames)} frames)")

    # 9. Save log
    log_path = os.path.join(GIF_OUTPUT_DIR, f"integration_rl_{plate_key}_log.json")
    log_output = {
        "controller": "RL (PPO)",
        "plate": plate_key,
        "total_time_s": total_time,
        "plants_inoculated": len(valid_tips),
        "avg_error_mm": avg_error * 1000,
        "max_error_mm": max_error * 1000,
        "converged": successes,
        "per_plant": results_log
    }
    with open(log_path, "w") as f:
        json.dump(log_output, f, indent=2)
    print(f"Log saved: {log_path}")

    sim.close()
    print("\nDone.")


if __name__ == "__main__":
    run_integration_rl()