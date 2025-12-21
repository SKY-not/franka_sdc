import numpy as np
from scipy.optimize import minimize
import math
import json
import os

# Franka Emika Panda DH Parameters (Modified DH)
# (a, d, alpha, theta_offset)
# Extracted from franka_emika.rs
# mdh_param!(theta_offset, d, a, alpha) -> (a, d, alpha, theta_offset)
# Note: The Rust macro seems to be (theta_offset, d, a, alpha) based on analysis.
# 1. mdh_param!(0., 0.333, 0., 0.)
# 2. mdh_param!(0., 0., 0., -FRAC_PI_2)
# 3. mdh_param!(0., 0.316, 0., FRAC_PI_2)
# 4. mdh_param!(0., 0., 0.0825, FRAC_PI_2)
# 5. mdh_param!(0., 0.384, -0.0825, -FRAC_PI_2)
# 6. mdh_param!(0., 0., 0., FRAC_PI_2)
# 7. mdh_param!(0., 0., 0.088, FRAC_PI_2)

DH_PARAMS = [
    # a,      d,      alpha,      theta_offset
    (0.0,    0.333,  0.0,        0.0),
    (0.0,    0.0,    -np.pi/2,   0.0),
    (0.0,    0.316,  np.pi/2,    0.0),
    (0.0825, 0.0,    np.pi/2,    0.0),
    (-0.0825,0.384,  -np.pi/2,   0.0),
    (0.0,    0.0,    np.pi/2,    0.0),
    (0.088,  0.0,    np.pi/2,    0.0),
]

# Joint limits (from franka_emika.rs)
JOINT_LIMITS = [
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
]

def mdh_transform(a, d, alpha, theta):
    """
    Calculates the Modified DH Transformation Matrix.
    T_{i-1, i} = Rot_x(alpha_{i-1}) * Trans_x(a_{i-1}) * Rot_z(theta_i) * Trans_z(d_i)
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct,        -st,       0,      a],
        [st * ca,   ct * ca,   -sa,    -d * sa],
        [st * sa,   ct * sa,   ca,     d * ca],
        [0,         0,         0,      1]
    ])

def forward_kinematics(joints):
    """
    Calculates the forward kinematics for the Franka Emika robot.
    Returns the homogeneous transformation matrix of the last frame.
    """
    T = np.eye(4)
    for i, (a, d, alpha, offset) in enumerate(DH_PARAMS):
        theta = joints[i] + offset
        T_i = mdh_transform(a, d, alpha, theta)
        T = T @ T_i
    return T

def rotation_error(R_current, R_target):
    """
    Calculates the rotation error between two rotation matrices.
    Using the angle of the difference rotation matrix.
    """
    R_diff = R_target @ R_current.T
    tr = np.trace(R_diff)
    # Clamp trace to [-1, 3] to avoid numerical issues with acos
    # For rotation matrix, trace is 1 + 2cos(theta), so range is [-1, 3]
    tr = np.clip(tr, -1.0, 3.0)
    theta = np.arccos((tr - 1) / 2)
    return theta

def inverse_kinematics(target_pose, initial_guess=None):
    """
    Calculates the inverse kinematics using numerical optimization.
    
    Args:
        target_pose: 4x4 homogeneous transformation matrix.
        initial_guess: Optional initial joint configuration (7,).
    
    Returns:
        joints: 7 joint angles.
    """
    if initial_guess is None:
        initial_guess = [0.0] * 7
        # Use middle of limits as a safer default?
        # initial_guess = [(low + high) / 2 for low, high in JOINT_LIMITS]

    def objective_function(joints):
        current_pose = forward_kinematics(joints)
        
        # Position error
        pos_error = np.linalg.norm(current_pose[:3, 3] - target_pose[:3, 3])
        
        # Orientation error
        rot_error = rotation_error(current_pose[:3, :3], target_pose[:3, :3])
        
        # Weighted sum
        return pos_error + 0.5 * rot_error

    # Bounds for the optimizer
    bounds = JOINT_LIMITS

    result = minimize(
        objective_function,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        tol=1e-6,
        options={'maxiter': 100}
    )

    if result.success:
        return result.x
    else:
        print("Optimization failed:", result.message)
        return result.x

if __name__ == "__main__":
    # Example Usage
    
    # 1. Define a target joint configuration to generate a target pose
    # target_joints_ground_truth = np.array([np.pi*2/3, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
    target_joints_ground_truth = np.array([-2.54552557,  2.16433802,  0.41517172,  0.84912696,  2.12946657, -1.08497437,-1.37358477])
    print(f"Ground Truth Joints: {target_joints_ground_truth}")
    
    # 2. Calculate FK to get the target pose
    target_pose = forward_kinematics(target_joints_ground_truth)
    
    alpha = np.pi/4
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    target_pose = np.array(
        [[ 1,  0,  0.0,  0.6],
         [ 0,  1,  0.0,  0.0],
         [ 0.0,  0.0,  1.0,  0],
         [ 0.0,  0.0,  0.0,  1.0]]
    )
    print("\nTarget Pose (T):")
    print(target_pose)
    
    # 3. Solve IK
    print("\nSolving IK...")
    # Start from a neutral position or random
    initial_guess = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]) 
    solved_joints = inverse_kinematics(target_pose, initial_guess)
    
    print(f"\nSolved Joints: {solved_joints}")
    
    # 4. Verify
    solved_pose = forward_kinematics(solved_joints)
    pos_diff = np.linalg.norm(solved_pose[:3, 3] - target_pose[:3, 3])
    print("\nSolved Pose (T):")
    print(solved_pose)
    print(f"\nPosition Error: {pos_diff:.6f}")
    
    print("\nDifference from Ground Truth (Note: Multiple solutions may exist):")
    print(solved_joints - target_joints_ground_truth)

    # Save result to JSON
    output_data = [{"Joint": solved_joints.tolist()}]
    
    # Determine output path
    # Try to find 'traj' directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    traj_dir = os.path.join(project_root, 'traj')
    
    if os.path.exists(traj_dir):
        output_path = os.path.join(traj_dir, 'ik_result.json')
    else:
        # Fallback to script directory if traj doesn't exist
        output_path = os.path.join(script_dir, 'ik_result.json')

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\nSaved IK result to {output_path}")
