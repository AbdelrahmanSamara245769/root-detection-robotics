# pid_controller.py
"""
Task 10 – OT-2 PID Cartesian Controller (Visualized)

- Uses three PID controllers (X, Y, Z)
- Allows settling before stopping
- Prints distance every 500 steps
- Stops only after stable convergence
- Generates Step Response Graphs for X, Y, Z (Grading Requirement)
"""

import os
import time
import numpy as np
import imageio
import matplotlib.pyplot as plt

from controllers.pid import PID
from project_paths import OUTPUT_DIR, enter_simulation_dir

enter_simulation_dir()
from sim_class import Simulation  # noqa: E402


# ---------------- CONFIG ----------------

TARGET = np.array([0.22, -0.15, 0.27])
DT = 0.01
MAX_STEPS = 3000
TOL = 0.001          # 1 mm
HOLD_STEPS = 10      # allow settling

os.makedirs(OUTPUT_DIR, exist_ok=True)
GIF_PATH = os.path.join(OUTPUT_DIR, "pid_target_reach.gif")
PLOT_PATH = os.path.join(OUTPUT_DIR, "pid_response_plot.png")
GIF_FPS = 20


# ---------------- HELPERS ----------------

def get_pipette_position(state):
    key = list(state.keys())[0]
    return np.array(state[key]["pipette_position"])


# ---------------- SETUP ----------------

sim = Simulation(num_agents=1, render=True, rgb_array=True)

pid_x = PID(kp=4.0, ki=0.0, kd=1.0, dt=DT, output_limit=0.3)
pid_y = PID(kp=4.0, ki=0.0, kd=1.0, dt=DT, output_limit=0.3)
pid_z = PID(kp=5.0, ki=0.5, kd=1.2, dt=DT, output_limit=0.3)

frames = []
error_history = []

# New: History lists for plotting data
history_x = []
history_y = []
history_z = []

success_counter = 0
final_step = None


# ---------------- CONTROL LOOP ----------------

start_time = time.time()

for step in range(1, MAX_STEPS + 1):

    # Read state
    state = sim.run([[0, 0, 0, 0]])
    pos = get_pipette_position(state)

    # Track positions for plotting
    history_x.append(pos[0])
    history_y.append(pos[1])
    history_z.append(pos[2])

    # Compute error
    error = TARGET - pos
    error_norm = float(np.linalg.norm(error))
    error_history.append(error_norm)

    # PID control
    action = [
        pid_x.step(error[0]),
        pid_y.step(error[1]),
        pid_z.step(error[2]),
        0
    ]

    sim.run([action])

    # Capture frame
    if sim.current_frame is not None:
        frames.append(sim.current_frame)

    # Progress output every 500 steps
    if step % 500 == 0:
        print(f"Step {step} | Distance to target: {error_norm:.4f}")

    # Convergence check (with settling)
    if error_norm < TOL:
        success_counter += 1
        if success_counter >= HOLD_STEPS:
            final_step = step
            print(f"\nTarget reached at step {step} with distance {error_norm:.4f}")
            break
    else:
        success_counter = 0

    time.sleep(DT)


# ---------------- FINAL METRICS ----------------

final_state = sim.run([[0, 0, 0, 0]])
final_pos = get_pipette_position(final_state)
final_error = TARGET - final_pos

metrics = {
    "target": TARGET.tolist(),
    "final_position": final_pos.tolist(),
    "final_error_xyz": final_error.tolist(),
    "final_error_norm": float(np.linalg.norm(final_error)),
    "max_error_norm": float(max(error_history)),
    "steps_taken": final_step if final_step is not None else MAX_STEPS,
    "success": float(np.linalg.norm(final_error)) < TOL,
    "runtime_sec": time.time() - start_time
}


# ---------------- OUTPUT ----------------

print("\n=== FINAL RESULT ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

imageio.mimsave(GIF_PATH, frames, fps=GIF_FPS)
print(f"\nGIF saved to: {GIF_PATH}")

sim.close()


# ---------------- VISUALIZATION ----------------

print(f"Generating Step Response Plot...")

# Create a figure with 3 stacked subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Generate time axis based on actual steps taken
time_axis = np.arange(len(history_x)) * DT

# Plot X Axis
ax1.plot(time_axis, history_x, label='Actual X', color='blue')
ax1.axhline(y=TARGET[0], color='red', linestyle='--', label=f'Target X ({TARGET[0]}m)')
ax1.set_ylabel('Position (m)')
ax1.set_title('X-Axis Step Response')
ax1.legend(loc='lower right')
ax1.grid(True)

# Plot Y Axis
ax2.plot(time_axis, history_y, label='Actual Y', color='green')
ax2.axhline(y=TARGET[1], color='red', linestyle='--', label=f'Target Y ({TARGET[1]}m)')
ax2.set_ylabel('Position (m)')
ax2.set_title('Y-Axis Step Response')
ax2.legend(loc='lower right')
ax2.grid(True)

# Plot Z Axis (Crucial for showing Integral action)
ax3.plot(time_axis, history_z, label='Actual Z', color='purple')
ax3.axhline(y=TARGET[2], color='red', linestyle='--', label=f'Target Z ({TARGET[2]}m)')
ax3.set_ylabel('Position (m)')
ax3.set_xlabel('Time (seconds)')
ax3.set_title('Z-Axis Step Response (Gravity Compensation)')
ax3.legend(loc='lower right')
ax3.grid(True)

plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"Graph saved to: {PLOT_PATH}")
# plt.show() # Uncomment for the window to pop up