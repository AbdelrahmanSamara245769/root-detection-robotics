"""
Task 09 – Robotics Environment Interaction
OT-2 Digital Twin workspace discovery and corner traversal
"""

import time
import os
import numpy as np

import imageio  # pip install imageio pillow
import pybullet as p

from project_paths import OUTPUT_DIR, enter_simulation_dir

enter_simulation_dir()
from sim_class import Simulation  # noqa: E402


# Helper functions
def get_pipette_position(state):
    """
    Extract pipette position from simulation state.
    """
    robot_key = list(state.keys())[0]
    return state[robot_key]["pipette_position"]


def capture_frame(width=960, height=540):
    """
    Capture a RGB frame from the current PyBullet GUI camera.
    Returns a (H, W, 3) uint8 numpy array.

    Notes:
    - Works best in PyBullet GUI mode.
    - Uses the current debug visualizer camera, so set your view before running.
    """
    cam = p.getDebugVisualizerCamera()
    view_matrix = cam[2]
    proj_matrix = cam[3]

    _, _, rgba, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    img = np.reshape(rgba, (height, width, 4))[:, :, :3]
    return img.astype(np.uint8)


def find_axis_limit(sim, axis, velocity, threshold=1e-5, max_steps=2000):
    """
    Move along one axis until position no longer changes.

    axis: 0 = x, 1 = y, 2 = z
    velocity: signed float
    """
    last_pos = None

    for _ in range(max_steps):
        actions = [[0.0, 0.0, 0.0, 0]]
        actions[0][axis] = velocity

        state = sim.run(actions)
        pos = get_pipette_position(state)

        if last_pos is not None and abs(pos[axis] - last_pos) < threshold:
            return pos[axis]

        last_pos = pos[axis]
        time.sleep(0.01)

    return last_pos


def move_to_target(sim, target, speed=0.05, frames=None, capture_every=2):
    """
    Incrementally move the pipette toward a Cartesian target.

    Added (non-invasive):
    - If frames is a list, captures and appends frames periodically.
    - capture_every controls capture frequency to keep GIF size reasonable.
    """
    step_count = 0

    while True:
        state = sim.run([[0.0, 0.0, 0.0, 0]])
        pos = get_pipette_position(state)

        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        dz = target[2] - pos[2]

        # Capture a frame periodically
        if frames is not None and (step_count % capture_every == 0):
            try:
                frames.append(capture_frame())
            except Exception:
                # Do not affect motion logic if capture fails
                pass

        if abs(dx) < 0.002 and abs(dy) < 0.002 and abs(dz) < 0.002:
            # Capture a final frame at the target
            if frames is not None:
                try:
                    frames.append(capture_frame())
                except Exception:
                    pass
            break

        actions = [[
            speed if dx > 0 else -speed,
            speed if dy > 0 else -speed,
            speed if dz > 0 else -speed,
            0
        ]]

        sim.run(actions)
        time.sleep(0.01)
        step_count += 1


# Main program
if __name__ == "__main__":

    sim = Simulation(num_agents=1)

    print("\nDetermining work envelope automatically...\n")

    # Discover X limits
    max_x = find_axis_limit(sim, axis=0, velocity=0.1)
    min_x = find_axis_limit(sim, axis=0, velocity=-0.1)

    # Discover Y limits
    max_y = find_axis_limit(sim, axis=1, velocity=0.1)
    min_y = find_axis_limit(sim, axis=1, velocity=-0.1)

    # Discover Z limits
    max_z = find_axis_limit(sim, axis=2, velocity=0.1)
    min_z = find_axis_limit(sim, axis=2, velocity=-0.1)

    print("Discovered work envelope:")
    print(f"X: [{min_x:.3f}, {max_x:.3f}]")
    print(f"Y: [{min_y:.3f}, {max_y:.3f}]")
    print(f"Z: [{min_z:.3f}, {max_z:.3f}]\n")

    # Define workspace corners
    corners = [
        (min_x, min_y, min_z),
        (min_x, min_y, max_z),
        (min_x, max_y, min_z),
        (min_x, max_y, max_z),
        (max_x, min_y, min_z),
        (max_x, min_y, max_z),
        (max_x, max_y, min_z),
        (max_x, max_y, max_z),
    ]

    print("Moving to all 8 workspace corners...\n")

    visited_corners = []

    # --- GIF recording additions (no change to motion logic) ---
    frames = []
    # Optional: capture a few initial frames before motion starts
    try:
        for _ in range(5):
            frames.append(capture_frame())
            time.sleep(0.01)
    except Exception:
        pass
    # -----------------------------------------------------------

    for i, corner in enumerate(corners):
        print(f"Corner {i + 1}: Target {corner}")

        # Motion unchanged; only additional argument to record frames
        move_to_target(sim, corner, frames=frames, capture_every=2)

        state = sim.run([[0.0, 0.0, 0.0, 0]])
        reached = get_pipette_position(state)
        print(f"Reached position: {reached}\n")

        visited_corners.append(reached)
        time.sleep(0.5)

        # Capture a few frames while paused at the corner
        try:
            for _ in range(8):
                frames.append(capture_frame())
                time.sleep(0.01)
        except Exception:
            pass

    print("Workspace exploration complete.\n")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    corners_path = os.path.join(OUTPUT_DIR, "workspace_corners.txt")
    with open(corners_path, "w") as f:
        f.write("OT-2 Workspace Corner Coordinates\n\n")
        f.write(f"X range: [{min_x:.3f}, {max_x:.3f}]\n")
        f.write(f"Y range: [{min_y:.3f}, {max_y:.3f}]\n")
        f.write(f"Z range: [{min_z:.3f}, {max_z:.3f}]\n\n")
        for i, pos in enumerate(visited_corners):
            f.write(
                f"Corner {i + 1}: "
                f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})\n"
            )

    print(f"Workspace envelope and corner coordinates saved to {corners_path}")

    # --- GIF save (added) ---
    gif_path = os.path.join(OUTPUT_DIR, "workspace_corners.gif")
    if len(frames) > 0:
        # duration = seconds per frame; increase if playback is too fast
        imageio.mimsave(gif_path, frames, duration=0.03)
        print(f"Saved GIF to {gif_path}")
    else:
        print("No frames captured; GIF not created.")