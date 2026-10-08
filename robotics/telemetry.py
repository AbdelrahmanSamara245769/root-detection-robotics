"""
Task 12 — Telemetry Recording
Runs the PID integration on one plate and records detailed telemetry:
- Per-step pipette position (x, y, z)
- Per-step error to target
- Per-step velocity commands
Generates step response plots for documentation.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# WORKING DIRECTORY SETUP
# ──────────────────────────────────────────────
import project_paths

project_paths.enter_simulation_dir()

from controllers.pid import PID
from sim_class import Simulation

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
PID_KP_XY = 4.0
PID_KI_XY = 0.0
PID_KD_XY = 1.0
PID_KP_Z = 5.0
PID_KI_Z = 0.5
PID_KD_Z = 1.2
PID_OUTPUT_LIMIT = 0.3
DT = 0.01
MAX_STEPS = 3000
TOL = 0.001
HOLD_STEPS = 10

CV_DATA_PATH = project_paths.CV_DATA_PATH
OUTPUT_DIR = os.path.join(project_paths.OUTPUT_DIR, "telemetry")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLATE_TO_KEY = {
    "test_image_10.png": "test_image_10",
    "test_image_14.png": "test_image_14",
    "test_image_16.png": "test_image_16",
}

COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
PLANT_LABELS = ['Plant 0', 'Plant 1', 'Plant 2', 'Plant 3', 'Plant 4']


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def get_pipette_position(state):
    key = list(state.keys())[0]
    return np.array(state[key]["pipette_position"])


def run_telemetry_for_plant(sim, target, plant_label):
    """Run PID to target and record full telemetry. Returns dict of arrays."""
    pid_x = PID(kp=PID_KP_XY, ki=PID_KI_XY, kd=PID_KD_XY, dt=DT, output_limit=PID_OUTPUT_LIMIT)
    pid_y = PID(kp=PID_KP_XY, ki=PID_KI_XY, kd=PID_KD_XY, dt=DT, output_limit=PID_OUTPUT_LIMIT)
    pid_z = PID(kp=PID_KP_Z, ki=PID_KI_Z, kd=PID_KD_Z, dt=DT, output_limit=PID_OUTPUT_LIMIT)

    log = {
        "time": [], "pos_x": [], "pos_y": [], "pos_z": [],
        "err_x": [], "err_y": [], "err_z": [], "err_norm": [],
        "vel_x": [], "vel_y": [], "vel_z": []
    }

    success_counter = 0
    converged_step = None

    for step in range(1, MAX_STEPS + 1):
        state = sim.run([[0, 0, 0, 0]])
        pos = get_pipette_position(state)
        error = target - pos
        error_norm = float(np.linalg.norm(error))

        vx = pid_x.step(error[0])
        vy = pid_y.step(error[1])
        vz = pid_z.step(error[2])

        # Record telemetry
        log["time"].append(step * DT)
        log["pos_x"].append(pos[0])
        log["pos_y"].append(pos[1])
        log["pos_z"].append(pos[2])
        log["err_x"].append(error[0])
        log["err_y"].append(error[1])
        log["err_z"].append(error[2])
        log["err_norm"].append(error_norm)
        log["vel_x"].append(vx)
        log["vel_y"].append(vy)
        log["vel_z"].append(vz)

        sim.run([[vx, vy, vz, 0]])

        if error_norm < TOL:
            success_counter += 1
            if success_counter >= HOLD_STEPS and converged_step is None:
                converged_step = step
                print(f"  {plant_label}: Converged at step {step} | Error: {error_norm:.4f}m")
                break
        else:
            success_counter = 0

    if converged_step is None:
        print(f"  {plant_label}: Did not converge | Final error: {log['err_norm'][-1]:.4f}m")

    # Drop inoculum
    sim.run([[0, 0, 0, 1]])

    # Convert lists to arrays
    for k in log:
        log[k] = np.array(log[k])

    log["target"] = target
    log["converged_step"] = converged_step
    return log


def plot_step_response(all_logs, plate_key):
    """Generate a 4-panel step response plot for all plants."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # Panel 1: X Position
    ax = axes[0]
    for i, log in enumerate(all_logs):
        ax.plot(log["time"], log["pos_x"], color=COLORS[i], linewidth=1.2, label=PLANT_LABELS[i])
        ax.axhline(y=log["target"][0], color=COLORS[i], linestyle='--', alpha=0.5, linewidth=0.8)
    ax.set_ylabel('X Position (m)')
    ax.set_title(f'PID Step Response — {plate_key}', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Y Position
    ax = axes[1]
    for i, log in enumerate(all_logs):
        ax.plot(log["time"], log["pos_y"], color=COLORS[i], linewidth=1.2, label=PLANT_LABELS[i])
        ax.axhline(y=log["target"][1], color=COLORS[i], linestyle='--', alpha=0.5, linewidth=0.8)
    ax.set_ylabel('Y Position (m)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Z Position
    ax = axes[2]
    for i, log in enumerate(all_logs):
        ax.plot(log["time"], log["pos_z"], color=COLORS[i], linewidth=1.2, label=PLANT_LABELS[i])
    ax.axhline(y=0.17, color='red', linestyle='--', alpha=0.7, linewidth=0.8, label='Target Z')
    ax.set_ylabel('Z Position (m)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Error Norm
    ax = axes[3]
    for i, log in enumerate(all_logs):
        ax.plot(log["time"], log["err_norm"] * 1000, color=COLORS[i], linewidth=1.2, label=PLANT_LABELS[i])
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=0.8, label='1mm threshold')
    ax.set_ylabel('Error (mm)')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"telemetry_{plate_key}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved: {plot_path}")


def plot_velocity_commands(all_logs, plate_key):
    """Generate velocity command plot for all plants."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    titles = ['X Velocity Commands', 'Y Velocity Commands', 'Z Velocity Commands']
    keys = ['vel_x', 'vel_y', 'vel_z']

    for panel, (ax, title, key) in enumerate(zip(axes, titles, keys)):
        for i, log in enumerate(all_logs):
            ax.plot(log["time"], log[key], color=COLORS[i], linewidth=0.8, alpha=0.8, label=PLANT_LABELS[i])
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title(title, fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel('Time (s)')
    plt.suptitle(f'PID Velocity Commands — {plate_key}', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"telemetry_velocity_{plate_key}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Velocity plot saved: {plot_path}")


def plot_per_plant_detail(all_logs, plate_key):
    """Generate individual per-plant convergence plots."""
    for i, log in enumerate(all_logs):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        target = log["target"]
        t = log["time"]

        # X
        axes[0, 0].plot(t, log["pos_x"], color=COLORS[i], linewidth=1.2)
        axes[0, 0].axhline(y=target[0], color='red', linestyle='--', linewidth=0.8)
        axes[0, 0].set_title(f'X Position (target: {target[0]:.4f}m)')
        axes[0, 0].set_ylabel('Position (m)')
        axes[0, 0].grid(True, alpha=0.3)

        # Y
        axes[0, 1].plot(t, log["pos_y"], color=COLORS[i], linewidth=1.2)
        axes[0, 1].axhline(y=target[1], color='red', linestyle='--', linewidth=0.8)
        axes[0, 1].set_title(f'Y Position (target: {target[1]:.4f}m)')
        axes[0, 1].set_ylabel('Position (m)')
        axes[0, 1].grid(True, alpha=0.3)

        # Z
        axes[1, 0].plot(t, log["pos_z"], color=COLORS[i], linewidth=1.2)
        axes[1, 0].axhline(y=target[2], color='red', linestyle='--', linewidth=0.8)
        axes[1, 0].set_title(f'Z Position (target: {target[2]:.4f}m)')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Position (m)')
        axes[1, 0].grid(True, alpha=0.3)

        # Error
        axes[1, 1].plot(t, log["err_norm"] * 1000, color=COLORS[i], linewidth=1.2)
        axes[1, 1].axhline(y=1.0, color='red', linestyle='--', linewidth=0.8)
        axes[1, 1].set_title('Error Norm')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Error (mm)')
        axes[1, 1].grid(True, alpha=0.3)

        converged = log["converged_step"]
        status = f"Converged at step {converged}" if converged else "Did not converge"
        final_err = log["err_norm"][-1] * 1000

        plt.suptitle(f'{plate_key} — Plant {i} | {status} | Final error: {final_err:.2f}mm',
                      fontsize=12, fontweight='bold')
        plt.tight_layout()

        plot_path = os.path.join(OUTPUT_DIR, f"telemetry_{plate_key}_plant_{i}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

    print(f"Per-plant plots saved ({len(all_logs)} plants)")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run_telemetry():
    print("=" * 60)
    print("TASK 12 — TELEMETRY RECORDING")
    print("=" * 60)

    # Load root tip data
    with open(CV_DATA_PATH, "r") as f:
        root_tip_data = json.load(f)

    # Start simulation
    print("\nInitializing simulation...")
    sim = Simulation(num_agents=1, render=False, rgb_array=False)

    plate_path = sim.get_plate_image()
    plate_filename = os.path.basename(plate_path)
    plate_key = PLATE_TO_KEY.get(plate_filename)

    if plate_key is None or plate_key not in root_tip_data:
        print(f"ERROR: Unrecognized plate '{plate_filename}'. Run again.")
        sim.close()
        return

    print(f"Plate: {plate_key}")

    tips = root_tip_data[plate_key]
    valid_tips = [(i, t) for i, t in enumerate(tips) if t is not None]
    print(f"Root tips: {len(valid_tips)}/5\n")

    # Run telemetry for each plant
    all_logs = []
    for lane, tip_data in valid_tips:
        target = np.array(tip_data["robot_coords_m"])
        print(f"Plant {lane}: target=({target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f})")
        log = run_telemetry_for_plant(sim, target, f"Plant {lane}")
        all_logs.append(log)

    sim.close()

    # Generate plots
    print("\nGenerating plots...")
    plot_step_response(all_logs, plate_key)
    plot_velocity_commands(all_logs, plate_key)
    plot_per_plant_detail(all_logs, plate_key)

    print(f"\nAll telemetry saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    run_telemetry()