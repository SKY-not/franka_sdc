import numpy as np
import sys

def calculate_velocity(start, target, T, g=9.81):
    """
    Calculate initial velocity vector given flight time T.
    
    Args:
        start (np.array): Starting position [x, y, z]
        target (np.array): Target position [x, y, z]
        T (float): Time of flight
        g (float): Gravity
        
    Returns:
        np.array: Initial velocity vector [vx, vy, vz]
    """
    if T <= 0:
        return np.zeros(3)
        
    delta = target - start
    
    # Horizontal velocity is constant
    vx = delta[0] / T
    vy = delta[1] / T
    
    # Vertical velocity: z = z0 + vz*t - 0.5*g*t^2
    # zt = z0 + vz*T - 0.5*g*T^2
    # vz*T = (zt - z0) + 0.5*g*T^2
    # vz = dz/T + 0.5*g*T
    vz = (delta[2] + 0.5 * g * T**2) / T
    
    return np.array([vx, vy, vz])

def get_min_energy_time(start, target, g=9.81):
    """
    Calculate the time of flight that minimizes the initial speed (Minimum Energy).
    
    Derivation:
    |v|^2 = (R^2 + dz^2)/T^2 + dz*g + 0.25*g^2*T^2
    d(|v|^2)/dT = -2(dist^2)/T^3 + 0.5*g^2*T = 0
    T^4 = 4 * dist^2 / g^2
    T = sqrt(2 * dist / g)
    """
    delta = target - start
    total_dist = np.linalg.norm(delta)
    
    # T for minimum velocity magnitude
    T_opt = np.sqrt(2 * total_dist / g)
    return T_opt

def solve_ballistic(start, target, balance_param=0.5, g=9.81):
    """
    Solve for ballistic trajectory.
    
    Args:
        start (list/array): Start pos
        target (list/array): Target pos
        balance_param (float): 0.0 to 1.0.
            0.0 = Minimum Energy (Slowest, High Arc)
            1.0 = High Speed (Faster, Flat Arc)
            
    Returns:
        dict: Result containing velocity, time, and speed.
    """
    start = np.array(start)
    target = np.array(target)
    
    # 1. Calculate the reference time (Minimum Energy Time)
    # This corresponds to the 45-degree launch (on flat ground)
    T_ref = get_min_energy_time(start, target, g)
    
    # 2. Determine actual time based on balance parameter
    # We want to reduce time as balance_param increases (to get flatter/faster shot)
    # Mapping:
    # balance 0.0 -> T = T_ref
    # balance 1.0 -> T = T_ref * 0.3 (Arbitrary lower bound, e.g., 3x faster)
    
    # Using an exponential decay or linear mapping
    # Let's use linear for simplicity
    min_time_factor = 0.3
    time_factor = 1.0 - (1.0 - min_time_factor) * balance_param
    
    T_sol = T_ref * time_factor
    
    # 3. Calculate Velocity
    v_sol = calculate_velocity(start, target, T_sol, g)
    speed = np.linalg.norm(v_sol)
    
    return {
        "velocity": v_sol,
        "time": T_sol,
        "speed": speed,
        "start": start,
        "target": target
    }

if __name__ == "__main__":
    # Example Usage
    start_pos = [0, 0, 0.5]
    target_pos = [2.0, 1.0, 0.0]
    
    print(f"Start: {start_pos}")
    print(f"Target: {target_pos}")
    print("-" * 30)
    
    # Test different balance parameters
    for balance in [0.0, 0.5, 0.9]:
        result = solve_ballistic(start_pos, target_pos, balance_param=balance)
        v = result['velocity']
        print(f"Balance: {balance:.1f}")
        print(f"  Time:  {result['time']:.4f} s")
        print(f"  Speed: {result['speed']:.4f} m/s")
        print(f"  Vel:   [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]")
        
        # Verify landing
        T = result['time']
        g = 9.81
        final_pos = start_pos + v * T - np.array([0, 0, 0.5 * g * T**2])
        err = np.linalg.norm(final_pos - target_pos)
        print(f"  Error: {err:.6f}")
        print("-" * 30)
