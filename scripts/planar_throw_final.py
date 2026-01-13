import numpy as np
import math
# from scipy.optimize import minimize
from HIK.ball_detector import BallDetector
import time
import sys
import os

# Ensure we can import franka_ik
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import franka_ik
except ImportError:
    # Fallback if running from root
    sys.path.append(os.path.join(os.getcwd(), 'scripts'))
    import franka_ik

import config

def calculate_T_acc(dq_release):
    """
    Calculates the acceleration time required to reach dq_release from rest.
    """
    v_max = config.JOINT_VEL_BOUND
    a_max = config.JOINT_ACC_BOUND
    j_max = config.JOINT_JERK_BOUND
    
    T_acc_req = np.zeros(7)
    for i in range(7):
        v = abs(dq_release[i])
        if v < 1e-6:
            continue
        t_a = 1.875 * v / a_max[i]
        t_j = np.sqrt(5.7735 * v / j_max[i])
        T_acc_req[i] = max(t_a, t_j)
        
    T_acc = np.max(T_acc_req)
    # Add some safety margin and ensure min time
    T_acc = max(T_acc * 1.5, 0.5)
    
    # Round T_acc to nearest dt
    dt = 0.001 # 1000Hz
    T_acc = np.ceil(T_acc / dt) * dt
    
    return T_acc

def solve_planar_throw(target_pos):
    """
    Calculates joint angles and velocities for a planar throw.
    
    Args:
        target_pos: [x, y, z] coordinates of the target.
        
    Returns:
        (q, dq): Tuple of joint angles (7,) and joint velocities (7,).
                 Returns None if no solution found.
    """
    target = np.array(target_pos)
    x, y, z_target = target
    
    # 1. Align Joint 1
    # q1 rotates around Z. 0 is X-axis.
    q1 = np.arctan2(y, x)
    
    # Check limits for q1
    limits = franka_ik.JOINT_LIMITS
    if not (limits[0][0] <= q1 <= limits[0][1]):
        print(f"Target angle {q1:.3f} rad is out of Joint 1 limits.")
        return None
        
    # Planar parameters
    r_target = np.sqrt(x**2 + y**2)
    g = 9.81
    
    # TCP Offset: Link8 (0.107) + GraspTarget (0.105)
    # The rotation of the hand (-45 deg) is around Z, so it doesn't affect the Z offset.
    tcp_offset_z = 0.212

    # Optimization
    # Variables: [q2, q4, q6, T]
    # q2, q4, q6 are the planar joints.
    # T is time of flight.
    
    # Initial guesses to try (Multi-start strategy)
    # Format: [q2, q4, q6, T]
    initial_guesses = [
        np.array([0.0, -1.5, 1.5, 0.5]),   # 1. Default: Neutral
        np.array([-0.5, -1.0, 1.0, 0.4]),  # 2. Cocked back: Faster throw
        np.array([0.5, -1.5, 0.5, 0.6]),   # 3. Forward release: Higher arc
        np.array([0.0, -1.0, 2.0, 0.3]),   # 4. Different elbow config
        np.array([-0.8, -0.8, 0.8, 0.4]),  # 5. More aggressive back swing
        np.array([0.2, -1.8, 1.8, 0.7]),   # 6. High lob
        np.array([-0.3, -1.2, 1.8, 0.5]),  # 7. New guess
        np.array([0.3, -1.6, 1.2, 0.6]),   # 8. New guess
        np.array([-0.6, -0.9, 1.5, 0.45]), # 9. New guess
        np.array([0.1, -1.4, 2.0, 0.55])   # 10. New guess
    ]
    
    # Bounds
    # q2, q4, q6 limits
    # T limits (0.1 to 2.0s)
    bounds = [
        limits[1],
        limits[3],
        limits[5],
        (0.1, 2.0)
    ]
    
    # Velocity limits for 2, 4, 6
    vel_limits = config.JOINT_VEL_BOUND[[1, 3, 5]]
    # Weight matrix for weighted pseudo-inverse (minimize normalized velocity)
    L_mat = np.diag(vel_limits)
    
    def compute_kinematics(params):
        q2, q4, q6, T = params
        
        # Construct full q
        q = np.zeros(7)
        q[0] = q1
        q[1] = q2
        q[2] = 0.0
        q[3] = q4
        q[4] = 0.0
        q[5] = q6
        q[6] = 0.0
        
        # FK
        T_mat = franka_ik.forward_kinematics(q)
        
        # TCP Position
        # TCP is shifted along Z axis of the last frame
        r_vec = T_mat[:3, 2] * tcp_offset_z
        pos = T_mat[:3, 3] + r_vec
        
        # Current planar pos
        r_rel = np.sqrt(pos[0]**2 + pos[1]**2)
        z_rel = pos[2]
        
        # Required planar velocity
        # r_target = r_rel + v_r * T
        # z_target = z_rel + v_z * T - 0.5 * g * T^2
        v_r = (r_target - r_rel) / T
        v_z = (z_target - z_rel + 0.5 * g * T**2) / T
        
        v_planar_req = np.array([v_r, v_z])
        
        # Jacobian
        J = franka_ik.calculate_jacobian(q)
        J_v_7 = J[:3, [1, 3, 5]] # Columns for 2, 4, 6
        J_w_7 = J[3:, [1, 3, 5]]
        
        # Adjust Jacobian for TCP
        # v_tcp = v_7 + w_7 x r_vec
        # J_v_tcp = J_v_7 + cross(J_w_7, r_vec)
        J_v = np.zeros_like(J_v_7)
        for k in range(3):
            J_v[:, k] = J_v_7[:, k] + np.cross(J_w_7[:, k], r_vec)
        
        # Project Jacobian to planar (r, z)
        # r direction is [cos(q1), sin(q1), 0]
        # z direction is [0, 0, 1]
        # J_proj = [ [cos q1, sin q1, 0] . J_col_i ]
        #          [ [0, 0, 1]       . J_col_i ]
        
        c1 = np.cos(q1)
        s1 = np.sin(q1)
        
        # Row 1: Radial contribution
        J_r = c1 * J_v[0, :] + s1 * J_v[1, :]
        # Row 2: Vertical contribution
        J_z = J_v[2, :]
        
        J_planar = np.vstack([J_r, J_z]) # 2x3 matrix
        
        # Solve for dq_planar: J_planar * dq = v_planar_req
        # Weighted least squares: min ||L^-1 dq||^2 s.t. J dq = v
        # Solution: dq = L^2 J^T (J L^2 J^T)^-1 v
        
        # J_weighted = J * L
        J_w = J_planar @ L_mat
        
        # dq_norm = pinv(J_w) * v
        try:
            dq_norm = np.linalg.pinv(J_w) @ v_planar_req
        except np.linalg.LinAlgError:
            dq_norm = np.array([1e6, 1e6, 1e6])
            
        dq_planar = L_mat @ dq_norm
        
        return q, dq_planar, r_rel, z_rel
        
    def objective(params):
        q, dq_planar, r_rel, z_rel = compute_kinematics(params)
        
        # Cost 1: Velocity magnitude (normalized)
        norm_vel = dq_planar / vel_limits
        cost_vel = np.sum(norm_vel**2)
        
        return cost_vel
        
    def constraint_vel(params):
        q, dq_planar, r_rel, z_rel = compute_kinematics(params)
        # All velocities must be within limits (factor 1.0, safety handled in limits)
        # We used safety factor in franka_ik, so we can use 1.0 here relative to that.
        # Return positive if valid
        return 0.98 - np.abs(dq_planar / vel_limits)
        
    def constraint_throw_direction(params):
        # Ensure we are throwing towards the target
        q, dq_planar, r_rel, z_rel = compute_kinematics(params)
        return r_target - r_rel - 0.2 # At least 20cm away

    def constraint_height(params):
        # Ensure release point is above ground
        q, dq_planar, r_rel, z_rel = compute_kinematics(params)
        return z_rel - 0.1 # Minimum 10cm height

    def constraint_start_height(params):
        # Ensure start point of trajectory is above ground
        q, dq_planar, r_rel, z_rel = compute_kinematics(params)
        
        # Construct full dq
        dq = np.zeros(7)
        dq[1] = dq_planar[0]
        dq[3] = dq_planar[1]
        dq[5] = dq_planar[2]
        
        T_acc = calculate_T_acc(dq)
        
        # q_start
        delta_q_acc = 0.5 * dq * T_acc
        q_start = q - delta_q_acc
        
        # Check FK
        T_mat = franka_ik.forward_kinematics(q_start)
        r_vec = T_mat[:3, 2] * tcp_offset_z
        pos = T_mat[:3, 3] + r_vec
        
        # Return margin (pos[2] - min_height)
        # We want pos[2] >= 0.05 (5cm safety)
        return pos[2] - 0.05
        
    cons = [
        {'type': 'ineq', 'fun': constraint_vel},
        {'type': 'ineq', 'fun': constraint_throw_direction},
        {'type': 'ineq', 'fun': constraint_height},
        {'type': 'ineq', 'fun': constraint_start_height}
    ]
    
    # Run optimization with multi-start
    best_res = None
    
    for i, x0 in enumerate(initial_guesses):
        # print(f"Trying initial guess {i+1}/{len(initial_guesses)}: {x0}")
        res = minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP', options={'ftol': 1e-4, 'maxiter': 100})
        
        if res.success:
            print(f"Optimization succeeded with guess #{i+1}")
            best_res = res
            break
        else:
            # Keep the best failed result just in case, or debug
            pass
            
    if best_res and best_res.success:
        res = best_res
        q, dq_planar, r_rel, z_rel = compute_kinematics(res.x)
        dq = np.zeros(7)
        dq[1] = dq_planar[0]
        dq[3] = dq_planar[1]
        dq[5] = dq_planar[2]
        
        # --- Calculate q7 for gripper orientation ---
        # We want the line connecting the fingers (y_hand) to be perpendicular to the release velocity.
        # 1. Calculate release velocity vector in base frame
        T_mat = franka_ik.forward_kinematics(q)
        r_vec = T_mat[:3, 2] * tcp_offset_z
        J = franka_ik.calculate_jacobian(q)
        J_v_7 = J[:3, :]
        J_w_7 = J[3:, :]
        
        # Full TCP Jacobian
        J_v_tcp = np.zeros_like(J_v_7)
        for k in range(7):
             J_v_tcp[:, k] = J_v_7[:, k] + np.cross(J_w_7[:, k], r_vec)
             
        v_release = J_v_tcp @ dq
        
        # 2. Transform v_release to Frame 7 (which is T_mat with q7=0)
        R_7 = T_mat[:3, :3]
        v_local = R_7.T @ v_release
        
        # 3. Solve for q7
        # We want the line connecting the fingers (y_hand) to be perpendicular to the release velocity.
        # This prevents the fingers from blocking the object path.
        # y_hand is perpendicular to v_release => x_hand is parallel to v_release.
        # x_hand_local angle is (q7 - pi/4).
        # So q7 - pi/4 = angle(v_local)
        # q7 = angle(v_local) + pi/4
        
        q7_candidates = []
        base_angle = np.arctan2(v_local[1], v_local[0]) + np.pi / 4
        
        # Candidate 1
        q7_1 = base_angle
        # Candidate 2 (180 deg flip)
        q7_2 = base_angle + np.pi
        
        for ang in [q7_1, q7_2]:
            # Normalize to [-pi, pi] first
            ang = (ang + np.pi) % (2 * np.pi) - np.pi
            # Check limits
            if limits[6][0] <= ang <= limits[6][1]:
                q7_candidates.append(ang)
                
        if q7_candidates:
            # Pick the one closest to 0
            q[6] = min(q7_candidates, key=abs)
            print(f"Calculated q7: {q[6]:.3f} rad")
        else:
            print("Warning: Could not find valid q7 for gripper orientation. Using 0.")
            
        print(f"Solution found:")
        print(f"Release Pos: r={r_rel:.3f}, z={z_rel:.3f}")
        print(f"Target: r={r_target:.3f}, z={z_target:.3f}")
        print(f"Flight Time: {res.x[3]:.3f}s")
        print(f"Joint Velocities: {dq_planar}")
        
        # Generate full trajectory (Accel -> Release -> Decel)
        full_traj, release_idx = generate_full_trajectory(q, dq)
        
        return {
            "q": q,
            "dq": dq,
            "t_flight": res.x[3],
            "release_pos": [r_rel * np.cos(q1), r_rel * np.sin(q1), z_rel],
            "trajectory": full_traj,
            "release_index": release_idx
        }
    else:
        print("Optimization failed for all initial guesses.")
        if best_res:
             print(f"Last error: {best_res.message}")
        return None

def generate_full_trajectory(q_release, dq_release):
    """
    Generates a trajectory that accelerates from 0 to dq_release and then decelerates to 0.
    Uses 5th order polynomial (S-curve velocity) to satisfy jerk limits.
    """
    # Limits
    v_max = config.JOINT_VEL_BOUND
    a_max = config.JOINT_ACC_BOUND
    j_max = config.JOINT_JERK_BOUND
    
    # 1. Calculate min time for each joint to reach dq_release from 0
    # Using S-curve velocity profile v(t) = v_target * S(t/T)
    # Max a = 1.875 * v / T
    # Max j = 5.7735 * v / T^2
    
    T_acc = calculate_T_acc(dq_release)
    dt = 0.001 # 1000Hz
    
    # 2. Calculate q_start
    # Delta q = 0.5 * v * T
    delta_q_acc = 0.5 * dq_release * T_acc
    q_start = q_release - delta_q_acc
    
    # 3. Calculate min time for deceleration (same logic)
    T_dec = T_acc 
    
    # 4. Calculate q_end
    delta_q_dec = 0.5 * dq_release * T_dec # Distance covered during decel
    q_end = q_release + delta_q_dec
    
    # Generate trajectory points
    traj_points = []
    
    # Acceleration Phase
    N_acc = int(round(T_acc / dt))
    for i in range(N_acc + 1): # Include end point
        t = i * dt
        # if t > T_acc: t = T_acc # No need to clamp if T_acc is multiple of dt
        tau = t / T_acc
        
        # Shape functions
        tau3 = tau**3
        tau4 = tau**4
        tau5 = tau**5
        tau6 = tau**6
        
        S = 6*tau5 - 15*tau4 + 10*tau3
        P = tau6 - 3*tau5 + 2.5*tau4
        
        # q = q_start + v_final * T * P(tau)
        q = q_start + dq_release * T_acc * P
        
        # v = v_final * S(tau)
        v = dq_release * S
        
        # a = v_final / T * S'(tau)
        dS = 30*tau4 - 60*tau3 + 30*(tau**2)
        a = dq_release / T_acc * dS
        
        # Check ground collision
        fk = franka_ik.forward_kinematics(q)
        if fk[2, 3] < 0.0:
            print(f"Warning: Ground collision detected at t={t:.3f} (Accel phase). z={fk[2,3]:.3f}")
        
        traj_points.append({
            "time": t,
            "q": q.tolist(),
            "dq": v.tolist(),
            "ddq": a.tolist()
        })
        
    # Deceleration Phase
    N_dec = int(round(T_dec / dt))
    for i in range(1, N_dec + 1): # Skip first point (duplicate of release)
        t_rel = i * dt
        t_total = T_acc + t_rel
        tau = t_rel / T_dec
        
        # We want to go from v to 0.
        tau3 = tau**3
        tau4 = tau**4
        tau5 = tau**5
        tau6 = tau**6
        
        S = 6*tau5 - 15*tau4 + 10*tau3
        P = tau6 - 3*tau5 + 2.5*tau4
        
        # q = q_release + v_release * T_dec * (tau - P)
        q = q_release + dq_release * T_dec * (tau - P)
        
        # v = v_release * (1 - S)
        v = dq_release * (1 - S)
        
        # a = v_release / T_dec * (-S')
        dS = 30*tau4 - 60*tau3 + 30*(tau**2)
        a = dq_release / T_dec * (-dS)
        
        # Check ground collision
        fk = franka_ik.forward_kinematics(q)
        if fk[2, 3] < 0.0:
            print(f"Warning: Ground collision detected at t={t_total:.3f} (Decel phase). z={fk[2,3]:.3f}")

        traj_points.append({
            "time": t_total,
            "q": q.tolist(),
            "dq": v.tolist(),
            "ddq": a.tolist()
        })
        
    return traj_points, N_acc

if __name__ == "__main__":
    # find cicle
    detector = BallDetector(
        arm_x=9*25,                    # 机械臂X坐标
        arm_z=-27*25,                  # 机械臂Z坐标
        camera_focal_length=6.0,       # 相机焦距
        pixel_size=0.0024,             # 像素尺寸
        ball_diameter=120.0            # 小球直径
    )
    detector.initialize_camera()
    x, y = detector.get_ball_position(display=False) 
    # Test case
    target = [x,y,0.0]
    print(target)
    target = [1.212, 0.153, 0.0]
    target = [1.128, -0.41, 0.0]
    dis = 1.2
    th = - np.pi / 12
    target = [dis*np.cos(th), dis*np.sin(th), 0]
    if len(sys.argv) > 3:
        target = [float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])]

    print(f"Solving for target: {target}")
    res = solve_planar_throw(target)
    if res:
        print("Result Q:", res["q"])
        print("Result dQ:", res["dq"])
        
        # Save to JSON
        import json
        traj_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'traj')
        os.makedirs(traj_dir, exist_ok=True)
        
        # 1. Save Plan (Summary)
        plan_path = os.path.join(traj_dir, 'throw_plan.json')
        # Convert numpy arrays to list
        output_data = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in res.items()}
        
        with open(plan_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Plan saved to {plan_path}")
        
        # 2. Save Full Trajectory (for Controller/Verification)
        # Format: [{"Joint": [q1, ..., q7]}, ...]
        full_traj_path = os.path.join(traj_dir, 'full_throw_trajectory.json')
        full_traj_data = [{"Joint": point["q"]} for point in res["trajectory"]]
        
        with open(full_traj_path, 'w') as f:
            json.dump(full_traj_data, f, indent=4)
        print(f"Full trajectory saved to {full_traj_path}")

        # 3. Save Gripper Trajectory
        gripper_path = os.path.join(traj_dir, 'gripper.json')
        rel_idx = res["release_index"] - 160
        total_len = len(full_traj_data)
        # True before release index, False at and after
        gripper_data = [True] * rel_idx + [False] * (total_len - rel_idx)
        
        with open(gripper_path, 'w') as f:
            json.dump(gripper_data, f, indent=4)
        print(f"Gripper trajectory saved to {gripper_path}")
        
        print("Starting Rust program...")   
        import subprocess
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # rust_command = ["cargo", "run", "--example", "sim_only_physics_franka"]
        rust_command = ["cargo", "run", "--release", "--example", "impedance_control_gripper"]
        
        # 选项 2: 运行主程序
        # rust_command = ["cargo", "run"]
        # 选项 3: 运行 release 版本（更快）
        # rust_command = ["cargo", "run", "--release", "--example", "sim_only_physics_franka"]
        
        try:
            # 在项目根目录执行 Rust 命令
            result = subprocess.run(
                rust_command,
                cwd=project_root,
                check=True,
                capture_output=False  # 设置为 False 以实时显示输出
            )
            print(f"\nRust program completed successfully (exit code: {result.returncode})")
        except subprocess.CalledProcessError as e:
            print(f"\nRust program failed with exit code: {e.returncode}")
        except FileNotFoundError:
            print("\nError: 'cargo' command not found. Make sure Rust is installed and in PATH.")
        except Exception as e:
            print(f"\nError running Rust program: {e}")

