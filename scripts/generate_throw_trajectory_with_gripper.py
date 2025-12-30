import numpy as np
import json
import os
import sys

# Add script dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
import franka_ik

def get_quintic_coeffs(q0, v0, a0, q1, v1, a1, T):
    T2 = T*T
    T3 = T2*T
    T4 = T3*T
    T5 = T4*T
    
    h = q1 - q0
    
    A = np.array([
        [T3, T4, T5],
        [3*T2, 4*T3, 5*T4],
        [6*T, 12*T2, 20*T3]
    ])
    b = np.array([
        h - v0*T - 0.5*a0*T2,
        v1 - v0 - a0*T,
        a1 - a0
    ])
    
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.zeros(6) # Should not happen for T > 0
    
    return np.array([q0, v0, 0.5*a0, x[0], x[1], x[2]])

def evaluate_quintic(coeffs, t):
    c0, c1, c2, c3, c4, c5 = coeffs
    
    t2 = t*t
    t3 = t2*t
    t4 = t3*t
    t5 = t4*t
    
    q = c0 + c1*t + c2*t2 + c3*t3 + c4*t4 + c5*t5
    v = c1 + 2*c2*t + 3*c3*t2 + 4*c4*t3 + 5*c5*t4
    a = 2*c2 + 6*c3*t + 12*c4*t2 + 20*c5*t3
    j = 6*c3 + 24*c4*t + 60*c5*t2
    
    return q, v, a, j

def check_limits(coeffs, T, v_max, a_max, j_max, q_min, q_max):
    # Diff-based check at control frequency (1000Hz)
    dt = 0.001
    # Generate time steps
    steps = int(np.ceil(T / dt)) + 2 
    times = np.arange(steps) * dt

    # Vectorized evaluation of q
    c0, c1, c2, c3, c4, c5 = coeffs
    
    t = times
    t2 = t*t
    t3 = t2*t
    t4 = t3*t
    t5 = t4*t
    
    q = c0 + c1*t + c2*t2 + c3*t3 + c4*t4 + c5*t5

    # Velocity (central diff using np.gradient which matches the manual implementation)
    v = np.gradient(q, dt)
    
    # Acceleration
    a = np.gradient(v, dt)
    
    # Jerk
    j = np.gradient(a, dt)

    # Strict constraint checks
    if np.any(q < q_min) or np.any(q > q_max):
        return False
    if np.any(np.abs(v) > v_max):
        return False
    if np.any(np.abs(a) > a_max):
        return False
    if np.any(np.abs(j) > j_max):
        return False

    return True

def find_min_duration(q0, v0, a0, q1, v1, a1, limits, q_limits):
    # limits: (v_max, a_max, j_max)
    # q_limits: (q_min, q_max)
    dist = abs(q1 - q0)
    
    # Heuristic start T
    T = 0.1
    if limits[0] > 0: T = max(T, dist / limits[0])
    if limits[1] > 0: T = max(T, np.sqrt(dist / limits[1]))
    if limits[2] > 0: T = max(T, np.power(dist / limits[2], 1/3))
    
    dt_search = 0.01 # Coarser search step for performance
    max_T = 30.0
    
    while T < max_T:
        coeffs = get_quintic_coeffs(q0, v0, a0, q1, v1, a1, T)
        if check_limits(coeffs, T, *limits, *q_limits):
            return T
        T += dt_search
        
    print(f"Warning: Could not find valid T within {max_T}s")
    return max_T

def check_cartesian_limits(coeffs, T):
    dt = 0.01
    steps = int(np.ceil(T / dt))
    
    for step in range(steps + 1):
        t = step * dt
        if t > T: t = T
        
        q_t = []
        for i in range(7):
            q, _, _, _ = evaluate_quintic(coeffs[i], t)
            q_t.append(q)
            
        T_ee = franka_ik.forward_kinematics(q_t)
        z = T_ee[2, 3]
        
        if z < 0.05:
            return False, t, z
            
    return True, 0, 0

def generate_trajectory():
    # 1. Load Plan
    traj_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'traj')
    plan_path = os.path.join(traj_dir, 'throw_plan.json')
    
    if not os.path.exists(plan_path):
        print(f"Error: {plan_path} not found. Run optimize_throw.py first.")
        return

    with open(plan_path, 'r') as f:
        plan = json.load(f)
        
    q_release = np.array(plan['q'])
    dq_release = np.array(plan['dq'])
    
    # 2. Define States
    # Initial State (Neutral)
    q_start = config.START_JOINT
    dq_start = np.zeros(7)
    ddq_start = np.zeros(7)
    
    # Release State
    # q_release, dq_release loaded
    ddq_release = np.zeros(7) # Assume 0 acceleration at release for smoothness
    
    # Final State (Stop Naturally)
    # Instead of returning to start, we calculate a braking position
    q_end = np.zeros(7)
    dq_end = np.zeros(7)
    ddq_end = np.zeros(7)
    
    # Limits
    v_limits = config.JOINT_VEL_BOUND
    a_limits = config.JOINT_ACC_BOUND
    j_limits = config.JOINT_JERK_BOUND
    
    # PLANNING SCALING: Use stricter limits for planning to ensure safety margin
    PLANNING_SCALE = config.SAFETY_FACTOR
    v_plan = v_limits * PLANNING_SCALE
    a_plan = a_limits * PLANNING_SCALE
    j_plan = j_limits * PLANNING_SCALE

    # Calculate Natural Stopping Position for Phase 2
    decel_factor = config.SAFETY_FACTOR
    for i in range(7):
        v = dq_release[i]
        if abs(v) < 1e-4:
            q_end[i] = q_release[i]
        else:
            # Estimate braking distance: d = v^2 / (2*a)
            # We use a heuristic to determine target position
            a_brake = a_limits[i] * decel_factor
            dist = (v**2) / (2 * a_brake) * np.sign(v)
            q_end[i] = q_release[i] + dist
            
        # Clamp to joint limits
        lower, upper = franka_ik.JOINT_LIMITS[i]
        q_end[i] = np.clip(q_end[i], lower + 0.05, upper - 0.05) # Add small margin

    # 3. Phase 1: Start -> Release
    print("Planning Phase 1: Approach...")
    T1_joints = []
    for i in range(7):
        # Adjust planning limits to accommodate boundary conditions
        v_local_limit = max(v_plan[i], abs(dq_start[i]), abs(dq_release[i]))
        a_local_limit = max(a_plan[i], abs(ddq_start[i]), abs(ddq_release[i]))
        
        # Ensure we don't exceed absolute hard limits
        v_local_limit = min(v_local_limit, v_limits[i])
        a_local_limit = min(a_local_limit, a_limits[i])

        T = find_min_duration(
            q_start[i], dq_start[i], ddq_start[i],
            q_release[i], dq_release[i], ddq_release[i],
            (v_local_limit, a_local_limit, j_plan[i]),
            franka_ik.JOINT_LIMITS[i]
        )
        T1_joints.append(T)
    
    T1 = max(T1_joints)
    # Round up to next ms to align with control frequency
    T1 = np.ceil(T1 * 1000) / 1000.0

    # Validate Phase 1 Cartesian limits
    print("Validating Phase 1 Cartesian limits...")
    while True:
        coeffs1 = []
        for i in range(7):
            c = get_quintic_coeffs(
                q_start[i], dq_start[i], ddq_start[i],
                q_release[i], dq_release[i], ddq_release[i],
                T1
            )
            coeffs1.append(c)
            
        valid, t_fail, z_fail = check_cartesian_limits(coeffs1, T1)
        if valid:
            break
            
        print(f"Phase 1 collision detected at t={t_fail:.3f}s (z={z_fail:.4f}m). Increasing T1...")
        T1 += 0.1
        if T1 > 30.0:
            print("Warning: Phase 1 duration exceeded 30s, proceeding with risk of collision.")
            break

    print(f"Phase 1 Duration: {T1:.4f} s")
    
    # 4. Phase 2: Release -> Stop (Return to Neutral)
    print("Planning Phase 2: Deceleration/Return...")
    T2_joints = []
    for i in range(7):
        # Adjust planning limits to accommodate boundary conditions
        v_local_limit = max(v_plan[i], abs(dq_release[i]), abs(dq_end[i]))
        a_local_limit = max(a_plan[i], abs(ddq_release[i]), abs(ddq_end[i]))
        
        v_local_limit = min(v_local_limit, v_limits[i])
        a_local_limit = min(a_local_limit, a_limits[i])

        T = find_min_duration(
            q_release[i], dq_release[i], ddq_release[i],
            q_end[i], dq_end[i], ddq_end[i],
            (v_local_limit, a_local_limit, j_plan[i]),
            franka_ik.JOINT_LIMITS[i]
        )
        T2_joints.append(T)
        
    T2 = max(T2_joints)
    # Round up to next ms
    T2 = np.ceil(T2 * 1000) / 1000.0

    # Validate Phase 2 Cartesian limits
    print("Validating Phase 2 Cartesian limits...")
    while True:
        coeffs2 = []
        for i in range(7):
            c = get_quintic_coeffs(
                q_release[i], dq_release[i], ddq_release[i],
                q_end[i], dq_end[i], ddq_end[i],
                T2
            )
            coeffs2.append(c)
            
        valid, t_fail, z_fail = check_cartesian_limits(coeffs2, T2)
        if valid:
            break
            
        print(f"Phase 2 collision detected at t={t_fail:.3f}s (z={z_fail:.4f}m). Increasing T2...")
        T2 += 0.1
        if T2 > 30.0:
            print("Warning: Phase 2 duration exceeded 30s, proceeding with risk of collision.")
            break

    print(f"Phase 2 Duration: {T2:.4f} s")
    
    # 5. Generate Full Trajectory
    freq = 1000.0
    dt = 1.0 / freq
    
    traj_data = []
    gripper_data = []
    
    # Phase 1 Generation
    coeffs1 = []
    for i in range(7):
        c = get_quintic_coeffs(
            q_start[i], dq_start[i], ddq_start[i],
            q_release[i], dq_release[i], ddq_release[i],
            T1
        )
        coeffs1.append(c)
        
    steps1 = int(np.ceil(T1 * freq))
    for step in range(steps1):
        t = step * dt
        q_t = []
        for i in range(7):
            q, _, _, _ = evaluate_quintic(coeffs1[i], t)
            q_t.append(q)
        # Gripper is CLOSED (True) during approach
        traj_data.append({"Joint": q_t})
        gripper_data.append(True)
        
    # Phase 2 Generation
    coeffs2 = []
    for i in range(7):
        c = get_quintic_coeffs(
            q_release[i], dq_release[i], ddq_release[i],
            q_end[i], dq_end[i], ddq_end[i],
            T2
        )
        coeffs2.append(c)
        
    steps2 = int(np.ceil(T2 * freq))
    for step in range(steps2):
        t = step * dt
        q_t = []
        for i in range(7):
            q, _, _, _ = evaluate_quintic(coeffs2[i], t)
            q_t.append(q)
        # Gripper is OPEN (False) during return/decel
        traj_data.append({"Joint": q_t})
        gripper_data.append(False)
        
    # Add final point explicitly to ensure we hit exactly
    traj_data.append({"Joint": q_end.tolist()})
    gripper_data.append(False)
    
    # 6. Save
    traj_output_path = os.path.join(traj_dir, 'full_throw_trajectory.json')
    with open(traj_output_path, 'w') as f:
        json.dump(traj_data, f, indent=4)
        
    gripper_output_path = os.path.join(traj_dir, 'gripper.json')
    with open(gripper_output_path, 'w') as f:
        json.dump(gripper_data, f, indent=4)
        
    print(f"Trajectory saved to {traj_output_path}")
    print(f"Gripper state saved to {gripper_output_path}")
    print(f"Total points: {len(traj_data)}")

if __name__ == "__main__":
    generate_trajectory()
