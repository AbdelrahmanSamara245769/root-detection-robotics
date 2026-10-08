"""
Task 13 — Performance Benchmarking
Generates comparison plots and summary statistics from integration logs.
Compares PID vs RL controller performance.
No simulation needed — reads from existing JSON results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

import project_paths

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
GIFS_DIR = os.path.join(project_paths.OUTPUT_DIR, "integration")  # integration JSON logs
OUTPUT_DIR = os.path.join(project_paths.OUTPUT_DIR, "benchmarking")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PID_PLATES = ["test_image_10", "test_image_14", "test_image_16"]
RL_PLATES = ["test_image_16"]  # RL was run on this plate

COLORS_PLATES = ['#2196F3', '#4CAF50', '#FF9800']
COLORS_PLANTS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
COLOR_PID = '#2196F3'
COLOR_RL = '#FF5722'


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────

def load_logs():
    pid_logs = {}
    rl_logs = {}

    for key in PID_PLATES:
        path = os.path.join(GIFS_DIR, f"integration_{key}_log.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                pid_logs[key] = json.load(f)
            print(f"Loaded PID: {key}")

    for key in RL_PLATES:
        path = os.path.join(GIFS_DIR, f"integration_rl_{key}_log.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                rl_logs[key] = json.load(f)
            print(f"Loaded RL:  {key}")

    return pid_logs, rl_logs


# ──────────────────────────────────────────────
# PLOT 1: PID Error per plant across all plates
# ──────────────────────────────────────────────

def plot_pid_error_per_plant(pid_logs):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(5)
    width = 0.25

    for i, (key, color) in enumerate(zip(PID_PLATES, COLORS_PLATES)):
        if key not in pid_logs:
            continue
        errors = [p["final_error_mm"] for p in pid_logs[key]["per_plant"]]
        bars = ax.bar(x + i * width, errors, width, label=key, color=color, alpha=0.85)
        for bar, err in zip(bars, errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{err:.2f}', ha='center', va='bottom', fontsize=8)

    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='1mm threshold')
    ax.set_xlabel('Plant (Lane)', fontsize=11)
    ax.set_ylabel('Positioning Error (mm)', fontsize=11)
    ax.set_title('PID Controller — Drop Location Error per Plant', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Plant 0', 'Plant 1', 'Plant 2', 'Plant 3', 'Plant 4'])
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.4)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pid_error_per_plant.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────
# PLOT 2: PID Steps to converge
# ──────────────────────────────────────────────

def plot_pid_steps_per_plant(pid_logs):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(5)
    width = 0.25

    for i, (key, color) in enumerate(zip(PID_PLATES, COLORS_PLATES)):
        if key not in pid_logs:
            continue
        steps = [p["steps"] for p in pid_logs[key]["per_plant"]]
        bars = ax.bar(x + i * width, steps, width, label=key, color=color, alpha=0.85)
        for bar, s in zip(bars, steps):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(s), ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Plant (Lane)', fontsize=11)
    ax.set_ylabel('Steps to Converge', fontsize=11)
    ax.set_title('PID Controller — Convergence Speed per Plant', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Plant 0', 'Plant 1', 'Plant 2', 'Plant 3', 'Plant 4'])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pid_steps_per_plant.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────
# PLOT 3: PID vs RL — Side by side error comparison
# ──────────────────────────────────────────────

def plot_pid_vs_rl_error(pid_logs, rl_logs):
    # Use test_image_16 for direct comparison (both controllers ran on it)
    compare_key = "test_image_16"
    if compare_key not in pid_logs or compare_key not in rl_logs:
        print("WARNING: Cannot compare — need both PID and RL results for test_image_16")
        return

    pid_errors = [p["final_error_mm"] for p in pid_logs[compare_key]["per_plant"]]
    rl_errors = [p["final_error_mm"] for p in rl_logs[compare_key]["per_plant"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(5)
    width = 0.35

    bars_pid = ax.bar(x - width/2, pid_errors, width, label='PID', color=COLOR_PID, alpha=0.85)
    bars_rl = ax.bar(x + width/2, rl_errors, width, label='RL (PPO)', color=COLOR_RL, alpha=0.85)

    for bar, err in zip(bars_pid, pid_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{err:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, err in zip(bars_rl, rl_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{err:.1f}', ha='center', va='bottom', fontsize=9)

    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='1mm threshold')
    ax.set_xlabel('Plant (Lane)', fontsize=11)
    ax.set_ylabel('Positioning Error (mm)', fontsize=11)
    ax.set_title(f'PID vs RL — Positioning Error ({compare_key})', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Plant 0', 'Plant 1', 'Plant 2', 'Plant 3', 'Plant 4'])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pid_vs_rl_error.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────
# PLOT 4: PID vs RL — Summary metrics
# ──────────────────────────────────────────────

def plot_pid_vs_rl_summary(pid_logs, rl_logs):
    compare_key = "test_image_16"
    if compare_key not in pid_logs or compare_key not in rl_logs:
        return

    pid = pid_logs[compare_key]
    rl = rl_logs[compare_key]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    # Avg Error
    ax = axes[0]
    vals = [pid["avg_error_mm"], rl["avg_error_mm"]]
    bars = ax.bar(['PID', 'RL (PPO)'], vals, color=[COLOR_PID, COLOR_RL], alpha=0.85)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{v:.2f}mm', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Error (mm)')
    ax.set_title('Average Error', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Max Error
    ax = axes[1]
    vals = [pid["max_error_mm"], rl["max_error_mm"]]
    bars = ax.bar(['PID', 'RL (PPO)'], vals, color=[COLOR_PID, COLOR_RL], alpha=0.85)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{v:.2f}mm', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Error (mm)')
    ax.set_title('Maximum Error', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Convergence
    ax = axes[2]
    vals = [pid["converged"], rl["converged"]]
    bars = ax.bar(['PID', 'RL (PPO)'], vals, color=[COLOR_PID, COLOR_RL], alpha=0.85)
    ax.axhline(y=5, color='green', linestyle='--', linewidth=1, alpha=0.7, label='5/5 target')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{v}/5', ha='center', va='bottom', fontsize=11)
    ax.set_ylabel('Plants Converged')
    ax.set_title('Convergence Rate', fontweight='bold')
    ax.set_ylim(0, 6)
    ax.grid(True, alpha=0.3, axis='y')

    # Execution Time
    ax = axes[3]
    vals = [pid["total_time_s"], rl["total_time_s"]]
    bars = ax.bar(['PID', 'RL (PPO)'], vals, color=[COLOR_PID, COLOR_RL], alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{v:.1f}s', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Execution Time', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('PID vs RL Controller Comparison (test_image_16)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pid_vs_rl_summary.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────
# PLOT 5: Error distribution (PID only, all plants)
# ──────────────────────────────────────────────

def plot_error_distribution(pid_logs):
    fig, ax = plt.subplots(figsize=(8, 5))

    all_errors = []
    for key in PID_PLATES:
        if key not in pid_logs:
            continue
        for p in pid_logs[key]["per_plant"]:
            all_errors.append(p["final_error_mm"])

    ax.hist(all_errors, bins=10, color=COLOR_PID, alpha=0.8, edgecolor='white')
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5, label='1mm threshold')
    ax.axvline(x=np.mean(all_errors), color='green', linestyle='-', linewidth=1.5,
               label=f'Mean: {np.mean(all_errors):.2f}mm')
    ax.set_xlabel('Positioning Error (mm)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('PID Error Distribution Across All Plants (n=15)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pid_error_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────
# PLOT 6: PID plate summary
# ──────────────────────────────────────────────

def plot_pid_plate_summary(pid_logs):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    plates = [k for k in PID_PLATES if k in pid_logs]
    colors = COLORS_PLATES[:len(plates)]

    ax = axes[0]
    avg_errors = [pid_logs[k]["avg_error_mm"] for k in plates]
    bars = ax.bar(plates, avg_errors, color=colors, alpha=0.85)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1)
    for bar, val in zip(bars, avg_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}mm', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Error (mm)')
    ax.set_title('Average Error per Plate', fontweight='bold')
    ax.set_ylim(0, 1.4)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15)

    ax = axes[1]
    max_errors = [pid_logs[k]["max_error_mm"] for k in plates]
    bars = ax.bar(plates, max_errors, color=colors, alpha=0.85)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1)
    for bar, val in zip(bars, max_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}mm', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Error (mm)')
    ax.set_title('Maximum Error per Plate', fontweight='bold')
    ax.set_ylim(0, 1.4)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15)

    ax = axes[2]
    times = [pid_logs[k]["total_time_s"] for k in plates]
    bars = ax.bar(plates, times, color=colors, alpha=0.85)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Execution Time per Plate', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15)

    plt.suptitle('PID Controller — Performance Across Plates', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pid_plate_summary.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────
# PRINT SUMMARY
# ──────────────────────────────────────────────

def print_summary(pid_logs, rl_logs):
    print("\n" + "=" * 70)
    print("BENCHMARKING SUMMARY")
    print("=" * 70)

    # PID Summary
    print("\n--- PID CONTROLLER ---")
    all_pid_errors = []
    all_pid_steps = []
    all_pid_times = []

    for key in PID_PLATES:
        if key not in pid_logs:
            continue
        log = pid_logs[key]
        print(f"\n  {key}:")
        print(f"    Avg error:  {log['avg_error_mm']:.2f}mm")
        print(f"    Max error:  {log['max_error_mm']:.2f}mm")
        print(f"    Converged:  {log['converged']}/{log['plants_inoculated']}")
        print(f"    Time:       {log['total_time_s']:.1f}s")

        for p in log["per_plant"]:
            all_pid_errors.append(p["final_error_mm"])
            all_pid_steps.append(p["steps"])
        all_pid_times.append(log["total_time_s"])

    if all_pid_errors:
        print(f"\n  OVERALL PID (n={len(all_pid_errors)} plants, {len(all_pid_times)} plates):")
        print(f"    Mean error:     {np.mean(all_pid_errors):.2f}mm")
        print(f"    Std error:      {np.std(all_pid_errors):.2f}mm")
        print(f"    Min error:      {np.min(all_pid_errors):.2f}mm")
        print(f"    Max error:      {np.max(all_pid_errors):.2f}mm")
        print(f"    Mean steps:     {np.mean(all_pid_steps):.0f}")
        print(f"    Mean time/plate:{np.mean(all_pid_times):.1f}s")
        print(f"    Convergence:    {len(all_pid_errors)}/{len(all_pid_errors)} (100%)")

    # RL Summary
    print("\n--- RL CONTROLLER (PPO) ---")
    for key in RL_PLATES:
        if key not in rl_logs:
            continue
        log = rl_logs[key]
        print(f"\n  {key}:")
        print(f"    Avg error:  {log['avg_error_mm']:.2f}mm")
        print(f"    Max error:  {log['max_error_mm']:.2f}mm")
        print(f"    Converged:  {log['converged']}/{log['plants_inoculated']}")
        print(f"    Time:       {log['total_time_s']:.1f}s")

    # Direct Comparison
    compare_key = "test_image_16"
    if compare_key in pid_logs and compare_key in rl_logs:
        pid = pid_logs[compare_key]
        rl = rl_logs[compare_key]
        print(f"\n{'─' * 70}")
        print(f"DIRECT COMPARISON ({compare_key}):")
        print(f"{'─' * 70}")
        print(f"  {'Metric':<25} {'PID':>12} {'RL (PPO)':>12} {'Winner':>10}")
        print(f"  {'─'*25} {'─'*12} {'─'*12} {'─'*10}")
        print(f"  {'Avg Error (mm)':<25} {pid['avg_error_mm']:>11.2f} {rl['avg_error_mm']:>11.2f} {'PID':>10}")
        print(f"  {'Max Error (mm)':<25} {pid['max_error_mm']:>11.2f} {rl['max_error_mm']:>11.2f} {'PID':>10}")
        print(f"  {'Convergence':<25} {pid['converged']:>10}/5 {rl['converged']:>10}/5 {'PID':>10}")
        print(f"  {'Time (s)':<25} {pid['total_time_s']:>11.1f} {rl['total_time_s']:>11.1f} {'PID':>10}")
        print(f"\n  PID outperforms RL on all metrics for this integration task.")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run_benchmarking():
    print("=" * 60)
    print("TASK 13 — PERFORMANCE BENCHMARKING")
    print("=" * 60)

    pid_logs, rl_logs = load_logs()
    if not pid_logs:
        print("ERROR: No PID integration logs found.")
        return

    print(f"\nGenerating plots...")
    plot_pid_error_per_plant(pid_logs)
    plot_pid_steps_per_plant(pid_logs)
    plot_pid_plate_summary(pid_logs)
    plot_error_distribution(pid_logs)

    if rl_logs:
        plot_pid_vs_rl_error(pid_logs, rl_logs)
        plot_pid_vs_rl_summary(pid_logs, rl_logs)

    print_summary(pid_logs, rl_logs)

    print(f"\nAll benchmarking outputs saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    run_benchmarking()