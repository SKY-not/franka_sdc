import json
import numpy as np
import os
import sys

# Add current directory to sys.path to ensure import works
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from franka_ik import inverse_kinematics, forward_kinematics
from throw_planner import search_feasible_throw
from config import SAFETY_FACTOR, JOINT_VEL_BOUND, JOINT_ACC_BOUND, JOINT_JERK_BOUND

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


def compute_quintic_coeffs(q0, qf, v0, vf, a0, af, T):
    """Return coefficients for a jerk-continuous quintic profile."""
    if T <= 0:
        raise ValueError("Trajectory duration must be positive")

    a0_coef = q0
    a1_coef = v0
    a2_coef = 0.5 * a0

    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T

    M = np.array([
        [T3,    T4,    T5],
        [3*T2,  4*T3,  5*T4],
        [6*T,  12*T2, 20*T3],
    ])

    rhs = np.array([
        qf - (a0_coef + a1_coef * T + a2_coef * T2),
        vf - (a1_coef + 2 * a2_coef * T),
        af - (2 * a2_coef),
    ])

    a3_coef, a4_coef, a5_coef = np.linalg.solve(M, rhs)
    return np.array([a0_coef, a1_coef, a2_coef, a3_coef, a4_coef, a5_coef])


def evaluate_quintic(coeffs, t):
    a0_coef, a1_coef, a2_coef, a3_coef, a4_coef, a5_coef = coeffs
    pos = (
        a0_coef
        + a1_coef * t
        + a2_coef * t**2
        + a3_coef * t**3
        + a4_coef * t**4
        + a5_coef * t**5
    )
    vel = (
        a1_coef
        + 2 * a2_coef * t
        + 3 * a3_coef * t**2
        + 4 * a4_coef * t**3
        + 5 * a5_coef * t**4
    )
    acc = (
        2 * a2_coef
        + 6 * a3_coef * t
        + 12 * a4_coef * t**2
        + 20 * a5_coef * t**3
    )
    jerk = (
        6 * a3_coef
        + 24 * a4_coef * t
        + 60 * a5_coef * t**2
    )
    return pos, vel, acc, jerk


def max_profile_values(coeffs, T, samples=200):
    """Sample the profile and return peak |vel|, |acc|, |jerk|."""
    ts = np.linspace(0.0, T, samples)
    max_vel = 0.0
    max_acc = 0.0
    max_jerk = 0.0
    for t in ts:
        _, vel, acc, jerk = evaluate_quintic(coeffs, t)
        max_vel = max(max_vel, abs(vel))
        max_acc = max(max_acc, abs(acc))
        max_jerk = max(max_jerk, abs(jerk))
    return max_vel, max_acc, max_jerk


def synchronize_profiles(q_start, q_target, dq_target, t_guess, dt):
    """Find a duration aligned to dt that satisfies v/a/jerk limits for all joints."""
    T = max(t_guess, dt)
    for _ in range(100):
        ticks = max(1, int(np.ceil(T / dt)))
        T_discrete = ticks * dt
        coeffs_list = []
        violated = False
        for axis in range(len(q_start)):
            try:
                coeffs = compute_quintic_coeffs(
                    q_start[axis],
                    q_target[axis],
                    0.0,
                    dq_target[axis],
                    0.0,
                    0.0,
                    T_discrete,
                )
            except np.linalg.LinAlgError:
                violated = True
                break
            vmax, amax, jmax = max_profile_values(coeffs, T_discrete)
            vel_limit = max(JOINT_VEL_BOUND[axis], abs(dq_target[axis]))
            if (
                vmax > vel_limit + 1e-6
                or amax > JOINT_ACC_BOUND[axis] + 1e-6
                or jmax > JOINT_JERK_BOUND[axis] + 1e-6
            ):
                violated = True
                break
            coeffs_list.append(coeffs)
        if not violated:
            return T_discrete, ticks, coeffs_list
        T = T_discrete * 1.05
    raise RuntimeError("Unable to satisfy jerk constraints with reasonable timing")

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

def generate_trajectory():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 1. Define Throw Task
    # Target is fixed, but release position will be searched
    ball_target_pos = [0.9, 0.2, 0.0] 
    
    # Search Parameters
    start_pos = [0.3, 0.3, 0.3]
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
    
    # 2. Calculate synchronized time with jerk constraints
    dists = q_target - q_start
    t_guess = max(
        get_min_time_rest_to_vel(dists[i], dq_target[i], JOINT_ACC_BOUND[i])
        for i in range(7)
    )

    dt = 1.0 / 240.0
    t_sync, ticks, coeffs_per_joint = synchronize_profiles(q_start, q_target, dq_target, t_guess, dt)

    print(f"Approach Time: {t_sync:.4f}s ({ticks} ticks)")

    # 3. Generate Approach Trajectory using jerk-limited quintic profiles
    traj_points = []

    for k in range(ticks + 1):
        t = min(k * dt, t_sync)
        point = []
        for axis in range(7):
            pos, _, _, _ = evaluate_quintic(coeffs_per_joint[axis], t)
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

if __name__ == "__main__":
    generate_trajectory()
