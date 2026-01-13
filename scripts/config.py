import numpy as np

SAFETY_FACTOR = 0.6
JOINT_VEL_BOUND = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]) * SAFETY_FACTOR
JOINT_ACC_BOUND = np.array([15., 7.5, 10., 12.5, 15., 20., 20.]) * SAFETY_FACTOR
JOINT_JERK_BOUND = np.array([7500., 3750., 5000., 6250., 7500., 10000., 10000.]) * SAFETY_FACTOR
DEFAULT_JOINT = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
START_JOINT = np.array([0.08897, -1.17644, -1.09569, -2.1661, -1.0026, 1.6852, 1.1039])