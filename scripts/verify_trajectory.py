import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Add script dir to path to import config and franka_ik
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
try:
    import franka_ik
except ImportError:
    # Fallback if running from root
    sys.path.append(os.path.join(os.getcwd(), 'scripts'))
    import franka_ik

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

    # 4. Check Ground Collision (TCP Position)
    print("\nChecking for ground collision...")
    tcp_positions = []
    tcp_offset_z = 0.212 # From planar_throw.py
    min_z = float('inf')
    collision_detected = False
    
    for k in range(n_points):
        T_mat = franka_ik.forward_kinematics(q[k])
        r_vec = T_mat[:3, 2] * tcp_offset_z
        pos = T_mat[:3, 3] + r_vec
        tcp_positions.append(pos)
        
        if pos[2] < min_z:
            min_z = pos[2]
        
        if pos[2] < 0.0:
            collision_detected = True
            
    tcp_positions = np.array(tcp_positions)
    
    print(f"Minimum TCP Height: {min_z:.4f} m")
    if collision_detected:
        print("FAILURE: Ground collision detected!")
        passed = False
    else:
        print("SUCCESS: No ground collision detected.")

    if passed:
        print("\nOVERALL SUCCESS: All constraints satisfied.")
    else:
        print("\nOVERALL FAILURE: Some constraints violated.")

    # 5. Plotting
    # Plot 3D Trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(tcp_positions[:, 0], tcp_positions[:, 1], tcp_positions[:, 2], label='TCP Trajectory', linewidth=2)
    
    # Plot start and end
    ax.scatter(tcp_positions[0, 0], tcp_positions[0, 1], tcp_positions[0, 2], c='g', marker='o', s=50, label='Start')
    ax.scatter(tcp_positions[-1, 0], tcp_positions[-1, 1], tcp_positions[-1, 2], c='r', marker='x', s=50, label='End')
    
    # Plot ground plane
    x_min, x_max = np.min(tcp_positions[:, 0]), np.max(tcp_positions[:, 0])
    y_min, y_max = np.min(tcp_positions[:, 1]), np.max(tcp_positions[:, 1])
    margin = 0.2
    xx, yy = np.meshgrid(np.linspace(x_min-margin, x_max+margin, 10), np.linspace(y_min-margin, y_max+margin, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('End-Effector Trajectory')
    ax.legend()
    
    plt.show()

if __name__ == "__main__":
    verify_trajectory()
