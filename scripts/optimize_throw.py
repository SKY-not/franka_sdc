import numpy as np
from scipy.optimize import minimize
import sys
import os
import json

# Add the current directory to sys.path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import franka_ik
import config

def solve_throw_configuration(target_position, initial_q=None):
    """
    Solves for the release joint configuration and velocities to throw an object to target_position.
    
    Args:
        target_position: [x, y, z] coordinates of the target.
        initial_q: Initial guess for joint angles (7,).
        
    Returns:
        result_dict: Dictionary containing 'q', 'dq', 't_flight', 'release_pos', 'release_vel'.
    """
    
    # Constants
    g = np.array([0, 0, -9.81])
    target_pos = np.array(target_position)
    
    # Transformation from Link 7 to Grasp Target
    # Link 7 -> Link 8 (Flange): Trans(0, 0, 0.107)
    # Link 8 -> Hand: RotZ(-45 deg)
    # Hand -> Grasp: Trans(0, 0, 0.105)
    # Total: Trans(0, 0, 0.212) * RotZ(-45 deg)
    alpha = -np.pi / 4
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    T_7_to_grasp = np.array([
        [ca, -sa, 0, 0],
        [sa, ca, 0, 0],
        [0, 0, 1, 0.212],
        [0, 0, 0, 1]
    ])
    
    # Bounds
    q_bounds = franka_ik.JOINT_LIMITS
    dq_bounds = [(-v, v) for v in config.JOINT_VEL_BOUND]
    t_bounds = [(0.1, 5.0)] # Flight time between 0.1s and 5s
    
    # Combined bounds for x = [q (7), dq (7), t (1)]
    bounds = q_bounds + dq_bounds + t_bounds
    
    # Initial guess
    if initial_q is None:
        # Use a neutral pose or the one from franka_ik main
        initial_q = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
        
    initial_dq = np.zeros(7)
    initial_t = np.array([1.0]) # 1 second flight time guess
    
    x0 = np.concatenate([initial_q, initial_dq, initial_t])
    
    def unpack_x(x):
        q = x[:7]
        dq = x[7:14]
        t = x[14]
        return q, dq, t
    
    def objective(x):
        q, dq, t = unpack_x(x)
        # Minimize joint velocities (energy)
        return np.sum(dq**2)
    
    def calculate_release_state(q, dq):
        # Forward Kinematics for Link 7
        T_7 = franka_ik.forward_kinematics(q)
        # Transform to Grasp Target
        T_grasp = T_7 @ T_7_to_grasp
        P_release = T_grasp[:3, 3]
        
        # Jacobian for Link 7
        J = franka_ik.calculate_jacobian(q)
        V_link7 = (J @ dq)[:3]
        W_link7 = (J @ dq)[3:]
        
        # Velocity at Grasp Target
        # v_grasp = v_link7 + w_link7 x r
        r = P_release - T_7[:3, 3]
        V_release = V_link7 + np.cross(W_link7, r)
        
        return P_release, V_release, T_grasp

    def ballistic_constraint(x):
        q, dq, t = unpack_x(x)
        P_release, V_release, _ = calculate_release_state(q, dq)
        
        # Projectile motion equation
        P_impact = P_release + V_release * t + 0.5 * g * t**2
        
        return P_impact - target_pos

    def orientation_constraint(x):
        q, dq, t = unpack_x(x)
        _, V_release, T_grasp = calculate_release_state(q, dq)
        
        # Y-axis of the hand frame (direction of finger movement)
        Y_hand = T_grasp[:3, 1]
        
        # Velocity should be perpendicular to finger movement to avoid collision
        return np.dot(V_release, Y_hand)

    # Constraints dictionary for SLSQP
    constraints = [
        {'type': 'eq', 'fun': ballistic_constraint},
        {'type': 'eq', 'fun': orientation_constraint}
    ]
    
    print(f"Optimization started for target: {target_pos}")
    
    # Optimization
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 200, 'ftol': 1e-6, 'disp': True}
    )
    
    if result.success:
        q_sol, dq_sol, t_sol = unpack_x(result.x)
        
        P_release, V_release, T_grasp = calculate_release_state(q_sol, dq_sol)
        
        print("\nOptimization Successful!")
        print(f"Flight Time: {t_sol:.4f} s")
        print(f"Release Position: {P_release}")
        print(f"Release Velocity: {V_release}")
        print(f"Joint Angles: {q_sol}")
        print(f"Joint Velocities: {dq_sol}")
        
        return {
            "q": q_sol.tolist(),
            "dq": dq_sol.tolist(),
            "t_flight": t_sol,
            "release_pos": P_release.tolist(),
            "release_vel": V_release.tolist()
        }
    else:
        print("Optimization Failed:", result.message)
        return None

if __name__ == "__main__":
    # Example Target
    target = [1.2, -0.2, 0.0] # x, y, z
    print(f"dist: {np.linalg.norm(target)}")
    
    # You can change the target here or pass it via command line args if needed
    if len(sys.argv) > 3:
        target = [float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])]
        
    solution = solve_throw_configuration(target)
    
    if solution:
        # Save to JSON
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'traj', 'throw_plan.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(solution, f, indent=4)
        print(f"Plan saved to {output_path}")
