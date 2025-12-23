import numpy as np
import os
import sys
import json

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from franka_ik import inverse_kinematics, calculate_jacobian, forward_kinematics
from ballistic_solver import solve_ballistic, calculate_velocity, get_min_energy_time

# Joint Velocity Limits (rad/s)
JOINT_VEL_LIMITS = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])

def plan_throw_task(start_pos, target_pos, release_orientation=None, balance_param=0.5, optimize_for_limits=True, verbose=True):
    """
    Plan a throwing task with strict joint velocity limit checks.
    Prioritizes finding ANY feasible solution over optimizing for time/speed.
    """
    start_pos = np.array(start_pos)
    target_pos = np.array(target_pos)
    
    # 1. Determine Release Pose (Position + Orientation)
    if release_orientation is None or isinstance(release_orientation, str) and release_orientation == 'auto':
        # Auto-align orientation with velocity direction
        # We need a reference velocity to align with. Use Min Energy velocity.
        T_ref = get_min_energy_time(start_pos, target_pos)
        v_ref = calculate_velocity(start_pos, target_pos, T_ref)
        v_dir = v_ref / np.linalg.norm(v_ref)
        
        z_axis = v_dir
        world_z = np.array([0, 0, 1])
        y_axis = np.cross(z_axis, world_z)
        if np.linalg.norm(y_axis) < 1e-3:
            y_axis = np.array([0, 1, 0])
        else:
            y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        release_orientation = np.column_stack((x_axis, y_axis, z_axis))
        
    target_pose = np.eye(4)
    target_pose[:3, :3] = release_orientation
    target_pose[:3, 3] = start_pos
    
    # 2. Solve Inverse Kinematics for Release Configuration
    if verbose: print("Solving IK for release pose...")
    # Initial guess: standard home or previous result
    q_guess = [0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]
    q_release = inverse_kinematics(target_pose, initial_guess=q_guess)
    if verbose: print(f"Release Joints: {q_release}")
    
    # Verify IK
    T_check = forward_kinematics(q_release)
    pos_err = np.linalg.norm(T_check[:3, 3] - start_pos)
    if verbose: print(f"IK Position Error: {pos_err:.6f}")
    
    # 3. Calculate Jacobian
    J = calculate_jacobian(q_release)
    
    # 4. Search for Feasible Velocity
    if verbose: print("Searching for feasible flight time to satisfy joint limits...")
    
    # Calculate Minimum Energy Time (Reference)
    T_ref = get_min_energy_time(start_pos, target_pos)
    
    # Search range: From fast (0.5x) to very slow/lob (5.0x)
    test_Ts = np.linspace(T_ref * 0.5, T_ref * 5.0, 200)
    
    best_solution = None
    min_max_ratio = float('inf')
    
    found_feasible = False
    
    for T in test_Ts:
        # Calculate Cartesian Velocity for this flight time
        v_test = calculate_velocity(start_pos, target_pos, T)
        
        # Map to Joint Velocity
        v_cart_test = np.zeros(6)
        v_cart_test[:3] = v_test
        
        dq_test = np.linalg.pinv(J) @ v_cart_test
        
        # Check limits
        ratios = np.abs(dq_test) / JOINT_VEL_LIMITS
        max_ratio = np.max(ratios)
        
        if max_ratio < min_max_ratio:
            min_max_ratio = max_ratio
            best_solution = {
                "time": T,
                "velocity": v_test,
                "dq": dq_test,
                "ratio": max_ratio
            }
            
        if max_ratio <= 1.0:
            if verbose: print(f"Found feasible solution! Time: {T:.4f}s, Max Ratio: {max_ratio:.2f}")
            found_feasible = True
            break 
            
    if not found_feasible:
        if verbose: 
            print(f"WARNING: Could not find feasible solution. Best Ratio: {min_max_ratio:.2f}")
            print("Using best available solution (will violate limits).")
    
    v_linear = best_solution['velocity']
    dq_release = best_solution['dq']
    T_final = best_solution['time']
    
    if verbose:
        print(f"Selected Flight Time: {T_final:.4f}s")
        print(f"Required Linear Velocity: {v_linear}")
        print(f"Joint Velocities: {dq_release}")
    
    ballistic_res = {
        "velocity": v_linear,
        "time": T_final,
        "speed": np.linalg.norm(v_linear),
        "ratio": min_max_ratio
    }
    
    return {
        "q_release": q_release,
        "dq_release": dq_release,
        "v_cart": np.concatenate([v_linear, [0,0,0]]),
        "jacobian": J,
        "ballistic_info": ballistic_res
    }

def search_feasible_throw(target_pos, start_pos=None, search_radius=[0.15, 0.15, 0.15], steps=[4, 4, 4]):
    """
    Search for a feasible release position around a nominal start position.
    
    Args:
        target_pos: Target position for the ball.
        start_pos: Center of the search volume (nominal release position). If None, uses a default.
        search_radius: [rx, ry, rz] radius to search around start_pos.
        steps: [nx, ny, nz] number of steps in each dimension.
    """
    if start_pos is None:
        start_pos = [0.45, 0.0, 0.5] # Default workspace center
        
    print(f"Searching for feasible release position to hit {target_pos}...")
    print(f"Search Center (start_pos): {start_pos}, Radius: {search_radius}")
    
    # Grid search
    xs = np.linspace(start_pos[0] - search_radius[0], start_pos[0] + search_radius[0], steps[0])
    ys = np.linspace(start_pos[1] - search_radius[1], start_pos[1] + search_radius[1], steps[1])
    zs = np.linspace(start_pos[2] - search_radius[2], start_pos[2] + search_radius[2], steps[2])
    
    best_plan = None
    min_ratio = float('inf')
    
    count = 0
    total_checks = len(xs) * len(ys) * len(zs)
    
    for x in xs:
        for y in ys:
            for z in zs:
                candidate_pos = [x, y, z]
                try:
                    # Use auto orientation
                    plan = plan_throw_task(candidate_pos, target_pos, release_orientation='auto', verbose=False)
                    ratio = plan['ballistic_info'].get('ratio', float('inf'))
                    
                    if ratio < min_ratio:
                        min_ratio = ratio
                        best_plan = plan
                        best_plan['start_pos'] = candidate_pos # Store pos
                        print(f"New best: Pos=[{x:.3f}, {y:.3f}, {z:.3f}], Ratio={ratio:.2f}")
                        
                    if min_ratio <= 1.0:
                        print("Found feasible solution!")
                        return best_plan
                except Exception as e:
                    pass
                count += 1
                
    if best_plan:
        print(f"Search finished. Best Ratio: {min_ratio:.2f}")
        return best_plan
    else:
        raise Exception("No solution found")

if __name__ == "__main__":
    # Example Usage
    start = [0.2, 0.2, 0.1]
    target = [0.4, 0.2, 0.3]
    
    result = plan_throw_task(start, target, balance_param=0.9)
    
    # Save result for other tools
    output_path = "./traj/throw_plan.json"
    
    # Convert numpy arrays to list for JSON
    save_data = {
        "q_release": result['q_release'].tolist(),
        "dq_release": result['dq_release'].tolist(),
        "v_cart": result['v_cart'].tolist()
    }
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=4)
        
    print(f"Plan saved to {output_path}")
