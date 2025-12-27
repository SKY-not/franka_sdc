import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add script dir to path to import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def verify_trajectory():
    # 1. Load Trajectory
    traj_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'traj', 'full_throw_trajectory.json')
    
    if not os.path.exists(traj_path):
        print(f"Error: {traj_path} not found.")
        return

    print(f"Loading trajectory from {traj_path}...")
    with open(traj_path, 'r') as f:
        data = json.load(f)
    
    # Extract joints: shape (N, 7)
    q = np.array([d['Joint'] for d in data])
    n_points, n_joints = q.shape
    dt = 0.001 # 1000 Hz
    time = np.arange(n_points) * dt
    
    print(f"Loaded {n_points} points ({n_points*dt:.2f} seconds).")

    # 2. Compute Derivatives (Using np.gradient for strict consistency)
    # Velocity
    v = np.gradient(q, dt, axis=0)
    
    # Acceleration
    a = np.gradient(v, dt, axis=0)
    
    # Jerk
    j = np.gradient(a, dt, axis=0)

    # 3. Check Limits
    v_lim = config.JOINT_VEL_BOUND
    a_lim = config.JOINT_ACC_BOUND
    j_lim = config.JOINT_JERK_BOUND
    
    passed = True
    
    print("\n" + "="*80)
    print(f"{'Joint':<6} | {'Type':<5} | {'Max Val':<10} | {'Limit':<10} | {'% Limit':<8} | {'Status'}")
    print("-" * 80)
    
    for i in range(n_joints):
        # Velocity
        v_max = np.max(np.abs(v[:, i]))
        v_ratio = v_max / v_lim[i] * 100
        v_status = "OK" if v_max <= v_lim[i] else "FAIL"
        if v_status == "FAIL": passed = False
        print(f"J{i+1:<5} | Vel   | {v_max:<10.4f} | {v_lim[i]:<10.4f} | {v_ratio:<7.1f}% | {v_status}")
        
        # Acceleration
        a_max = np.max(np.abs(a[:, i]))
        a_ratio = a_max / a_lim[i] * 100
        a_status = "OK" if a_max <= a_lim[i] else "FAIL"
        if a_status == "FAIL": passed = False
        print(f"J{i+1:<5} | Acc   | {a_max:<10.4f} | {a_lim[i]:<10.4f} | {a_ratio:<7.1f}% | {a_status}")
        
        # Jerk
        j_max = np.max(np.abs(j[:, i]))
        j_ratio = j_max / j_lim[i] * 100
        j_status = "OK" if j_max <= j_lim[i] else "FAIL"
        if j_status == "FAIL": passed = False
        print(f"J{i+1:<5} | Jerk  | {j_max:<10.4f} | {j_lim[i]:<10.4f} | {j_ratio:<7.1f}% | {j_status}")
        print("-" * 80)

    if passed:
        print("\nSUCCESS: All constraints satisfied.")
    else:
        print("\nFAILURE: Some constraints violated.")

    # 4. Plotting - Separate plots for V, A, J
    # We will create 3 figures, each with 7 subplots
    
    # Plot Velocity
    fig_v, axes_v = plt.subplots(n_joints, 1, figsize=(10, 15), sharex=True)
    fig_v.suptitle('Joint Velocities', fontsize=16)
    for i in range(n_joints):
        axes_v[i].plot(time, v[:, i], label=f'J{i+1}')
        axes_v[i].axhline(y=v_lim[i], color='r', linestyle='--', alpha=0.5)
        axes_v[i].axhline(y=-v_lim[i], color='r', linestyle='--', alpha=0.5)
        axes_v[i].set_ylabel(f'J{i+1} (rad/s)')
        axes_v[i].grid(True)
        axes_v[i].legend(loc='upper right')
    axes_v[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Plot Acceleration
    fig_a, axes_a = plt.subplots(n_joints, 1, figsize=(10, 15), sharex=True)
    fig_a.suptitle('Joint Accelerations', fontsize=16)
    for i in range(n_joints):
        axes_a[i].plot(time, a[:, i], label=f'J{i+1}', color='orange')
        axes_a[i].axhline(y=a_lim[i], color='r', linestyle='--', alpha=0.5)
        axes_a[i].axhline(y=-a_lim[i], color='r', linestyle='--', alpha=0.5)
        axes_a[i].set_ylabel(f'J{i+1} (rad/s^2)')
        axes_a[i].grid(True)
        axes_a[i].legend(loc='upper right')
    axes_a[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Plot Jerk
    fig_j, axes_j = plt.subplots(n_joints, 1, figsize=(10, 15), sharex=True)
    fig_j.suptitle('Joint Jerks', fontsize=16)
    for i in range(n_joints):
        axes_j[i].plot(time, j[:, i], label=f'J{i+1}', color='green')
        axes_j[i].axhline(y=j_lim[i], color='r', linestyle='--', alpha=0.5)
        axes_j[i].axhline(y=-j_lim[i], color='r', linestyle='--', alpha=0.5)
        axes_j[i].set_ylabel(f'J{i+1} (rad/s^3)')
        axes_j[i].grid(True)
        axes_j[i].legend(loc='upper right')
    axes_j[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

if __name__ == "__main__":
    verify_trajectory()
