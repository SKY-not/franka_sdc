import json
import numpy as np
import os
import sys

# Add current directory to sys.path to ensure import works
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from franka_ik import inverse_kinematics, forward_kinematics
from throw_planner import plan_throw_task, search_feasible_throw
from ballistic_solver import solve_ballistic

# Safety Factor (0.0 < k <= 1.0)
# Adjust this value to limit velocity, acceleration and jerk for safety
# For throwing task, we need high velocity, so we increase safety factor or ignore it for the throw phase
SAFETY_FACTOR = 1.0
JOINT_VEL_BOUND = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]) * SAFETY_FACTOR
JOINT_ACC_BOUND = np.array([15., 7.5, 10., 12.5, 15., 20., 20.]) * SAFETY_FACTOR

# Default Joint Position (Start)
JOINT_DEFAULT = np.array([
    0.,
    -np.pi/4,
    0.,
    -3. * np.pi/4,
    0.,
    np.pi/2,
    np.pi/4,
])

def get_min_time_rest_to_vel(dist, vf, a_max):
    """
    Calculate min time to go from rest to velocity vf covering distance dist.
    Using Bang-Bang acceleration (either Backswing or Peak-Over).
    """
    # Normalize direction
    if vf < 0:
        dist = -dist
        vf = -vf
        
    # Now vf > 0.
    # Threshold distance for simple acceleration: d = 0.5 * vf^2 / a
    # Actually, min time is when we just accelerate to vf?
    # t = vf/a, d = 0.5 vf^2/a.
    # If dist == 0.5 vf^2/a, T = vf/a.
    
    d_threshold = 0.5 * vf**2 / a_max
    
    if dist < d_threshold:
        # Backswing needed (-a, +a)
        # t1^2 = vf^2/(2a^2) - dist/a
        term = vf**2 / (2 * a_max**2) - dist / a_max
        if term < 0: term = 0 # Should not happen if dist < d_threshold
        t1 = np.sqrt(term)
        # T = 2*t1 + vf/a
        T = 2 * t1 + vf / a_max
    else:
        # Peak-Over needed (+a, -a)
        # Or just +a if dist is huge?
        # For (+a, -a):
        # Quadratic for a was derived for fixed T.
        # Here we want min T with fixed a.
        # t1 - t2 = vf/a
        # 0.5 a t1^2 + a t1 t2 - 0.5 a t2^2 = dist
        # ...
        # Resulting T:
        # T^2 = (2 vf^2/a^2 + 4(dist - vf^2/a)/a )? No.
        # Let's use the quadratic for a in reverse.
        # T^2 a^2 - (4D - 2 T vf) a - vf^2 = 0
        # We want T.
        # a^2 T^2 + 2 vf a T - (4 D a + vf^2) = 0
        # Quadratic in T:
        # (a^2) T^2 + (2 vf a) T - (4 D a + vf^2) = 0
        
        A = a_max**2
        B = 2 * vf * a_max
        C = -(4 * dist * a_max + vf**2)
        
        # T = (-B + sqrt(B^2 - 4AC)) / 2A
        delta = B**2 - 4 * A * C
        T = (-B + np.sqrt(delta)) / (2 * A)
        
    return T

def solve_profile_rest_to_vel(dist, vf, T):
    """
    Solve for acceleration a and profile type given T, dist, vf.
    Returns params dict.
    """
    # Normalize
    sign = 1.0
    if vf < 0:
        dist = -dist
        vf = -vf
        sign = -1.0
        
    # Check threshold
    d_threshold = 0.5 * vf * T
    
    if dist < d_threshold:
        # Backswing (-a, +a)
        # Quadratic for a: T^2 a^2 + (4D - 2 T vf) a - vf^2 = 0
        A = T**2
        B = 4 * dist - 2 * T * vf
        C = -vf**2
        
        delta = B**2 - 4 * A * C
        a = (-B + np.sqrt(delta)) / (2 * A)
        
        # Calculate switching times
        # t1^2 = vf^2/(2a^2) - dist/a
        t1 = np.sqrt(vf**2 / (2 * a**2) - dist / a)
        t2 = T - t1
        
        return {
            'type': 'backswing',
            'a': a * sign, # Magnitude a, direction depends on sign
            't1': t1,
            't2': t2,
            'sign': sign
        }
    else:
        # Peak-Over (+a, -a)
        # Quadratic for a: T^2 a^2 - (4D - 2 T vf) a - vf^2 = 0
        A = T**2
        B = -(4 * dist - 2 * T * vf)
        C = -vf**2
        
        delta = B**2 - 4 * A * C
        a = (-B + np.sqrt(delta)) / (2 * A)
        
        # Calculate switching times
        # t1 = 0.5(T + vf/a)
        t1 = 0.5 * (T + vf / a)
        t2 = T - t1
        
        return {
            'type': 'peak_over',
            'a': a * sign,
            't1': t1,
            't2': t2,
            'sign': sign
        }

def generate_trajectory():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 1. Define Throw Task
    # Target is fixed, but release position will be searched
    ball_target_pos = [0.9, 0.0, 0.0] 
    
    # Search Parameters
    start_pos = [0.2, 0.3, 0.5]
    search_radius = [0.2, 0.3, 0.2] # Widened search radius
    
    print(f"Searching for feasible throw to target {ball_target_pos}...")
    print(f"Search Center (start_pos): {start_pos}, Radius: {search_radius}")
    
    # Use the new search function
    throw_plan = search_feasible_throw(ball_target_pos, start_pos=start_pos, search_radius=search_radius)
    
    q_target = throw_plan['q_release']
    dq_target = throw_plan['dq_release']
    
    # Get the found start position
    ball_start_pos = throw_plan.get('start_pos', [0,0,0]) # Fallback if not set
    
    print(f"Found feasible release position: {ball_start_pos}")
    print(f"Target Joints: {q_target}")
    print(f"Target Velocity: {dq_target}")
    print(f"Safety Factor: {SAFETY_FACTOR}")
    
    q_start = JOINT_DEFAULT
    dq_target = throw_plan['dq_release']
    
    print(f"Target Joints: {q_target}")
    print(f"Target Velocity: {dq_target}")
    print(f"Safety Factor: {SAFETY_FACTOR}")
    
    q_start = JOINT_DEFAULT
    
    # 2. Calculate Synchronized Time for Approach (Rest to Velocity)
    dists = q_target - q_start # Signed distance
    t_mins = []
    for i in range(7):
        t = get_min_time_rest_to_vel(dists[i], dq_target[i], JOINT_ACC_BOUND[i])
        t_mins.append(t)
        
    t_sync = max(t_mins)
    
    # Round up to nearest tick
    dt = 1.0 / 240.0
    ticks = int(np.ceil(t_sync / dt))
    if ticks == 0: ticks = 1
    t_sync = ticks * dt
    
    print(f"Approach Time: {t_sync:.4f}s ({ticks} ticks)")
    
    # 3. Generate Approach Trajectory
    traj_points = []
    
    # Calculate params for each joint
    params = []
    for i in range(7):
        p = solve_profile_rest_to_vel(dists[i], dq_target[i], t_sync)
        p['start'] = q_start[i]
        params.append(p)
        
    # Refined Evaluation Loop
    # Let's pre-calculate a1, a2 for each joint
    for p in params:
        a_mag = abs(p['a'])
        sign = p['sign']
        if p['type'] == 'backswing':
            # Flipped frame: -a, +a
            # Real frame: -sign*a, +sign*a
            p['a1'] = -sign * a_mag
            p['a2'] = sign * a_mag
        else:
            # Flipped frame: +a, -a
            # Real frame: +sign*a, -sign*a
            p['a1'] = sign * a_mag
            p['a2'] = -sign * a_mag
            
    for k in range(ticks + 1):
        t = k * dt
        point = []
        for i in range(7):
            p = params[i]
            if t <= p['t1']:
                # Phase 1
                # q = q0 + 0.5 * a1 * t^2
                pos = p['start'] + 0.5 * p['a1'] * t**2
            else:
                # Phase 2
                # q = q(t1) + v(t1)*dt + 0.5 * a2 * dt^2
                dt2 = t - p['t1']
                q_t1 = p['start'] + 0.5 * p['a1'] * p['t1']**2
                v_t1 = p['a1'] * p['t1']
                pos = q_t1 + v_t1 * dt2 + 0.5 * p['a2'] * dt2**2
            point.append(pos)
        traj_points.append({"Joint": point})
        
    # 4. Buffer Phase (Cartesian Follow-Through)
    # Instead of stopping joints immediately, continue moving along the Cartesian velocity vector
    # while decelerating to a stop. This prevents "pulling back" or "jerking" at release.
    
    print("Generating Cartesian Follow-Through...")
    
    # Get Release Pose (T_release)
    T_release = forward_kinematics(q_target)
    P_release = T_release[:3, 3]
    R_release = T_release[:3, :3]
    
    # Get Cartesian Velocity Vector
    v_cart_vec = throw_plan['v_cart'][:3]
    v_mag = np.linalg.norm(v_cart_vec)
    v_dir = v_cart_vec / v_mag
    
    # Deceleration parameters
    a_cart_dec = 5.0 # m/s^2 (Cartesian deceleration)
    t_stop = v_mag / a_cart_dec
    
    buffer_ticks = int(np.ceil(t_stop / dt))
    if buffer_ticks == 0: buffer_ticks = 1
    
    print(f"Follow-Through Time: {t_stop:.4f}s ({buffer_ticks} ticks)")
    
    q_prev = q_target
    
    for k in range(1, buffer_ticks + 1):
        t = k * dt
        if t > t_stop: t = t_stop
        
        # P(t) = P0 + v0*t - 0.5*a*t^2 (along v_dir)
        # dist = v0*t - 0.5 * (v0/t_stop) * t^2
        dist = v_mag * t - 0.5 * a_cart_dec * t**2
        
        P_curr = P_release + v_dir * dist
        
        # Construct Target Pose
        T_curr = np.eye(4)
        T_curr[:3, :3] = R_release # Keep orientation constant
        T_curr[:3, 3] = P_curr
        
        # Solve IK
        # Use previous joint angles as guess to ensure continuity
        q_curr = inverse_kinematics(T_curr, initial_guess=q_prev)
        
        traj_points.append({"Joint": q_curr.tolist()})
        q_prev = q_curr

    # Save
    out_path = os.path.join(project_root, 'traj', 'trajectory.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(traj_points, f, indent=4)
        
    print(f"Saved trajectory to {out_path}")
        
    print(f"Saved trajectory to {out_path}")

if __name__ == "__main__":
    generate_trajectory()
