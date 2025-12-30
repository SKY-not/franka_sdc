use anyhow::Result;
use franka_rust::{FrankaEmika, FrankaGripper, types::robot_types::SetCollisionBehaviorData};
use robot_behavior::{MotionType, behavior::*};
use std::{fs::File, thread::sleep, time::Duration};

fn main() -> Result<()> {
    // 初始化机器人franka
    let mut robot = FrankaEmika::new("172.16.0.2");
    robot.set_default_behavior()?;
    robot.set_collision_behavior(SetCollisionBehaviorData {
        lower_torque_thresholds_acceleration: [10.0; 7],
        upper_torque_thresholds_acceleration: [100.0; 7],
        lower_torque_thresholds_nominal: [10.0; 7],
        upper_torque_thresholds_nominal: [100.0; 7],
        lower_force_thresholds_acceleration: [10.0; 6],
        upper_force_thresholds_acceleration: [100.0; 6],
        lower_force_thresholds_nominal: [10.0; 6],
        upper_force_thresholds_nominal: [100.0; 6],
    })?;
    // 初始化对应的夹爪
    let mut gripper = FrankaGripper::new("172.16.0.2");
    gripper.homing().unwrap();
    println!("Gripper initialized!");

    // 加载机械臂轨迹
    let file = File::open("./traj/full_throw_trajectory.json")?;
    let traj: Vec<MotionType<7>> = serde_json::from_reader(file)?;
    // 加载夹爪轨迹
    let file = File::open("./traj/gripper.json")?;
    let gripper_traj: Vec<bool> = serde_json::from_reader(file)?;
    // 夹爪与机械臂轨迹点数应相同
    assert_eq!(traj.len(), gripper_traj.len());

    // 运动到抓取位置
    // TODO: 添加一行代码，运动到抓取位置,然后夹取物体

    // 运动到轨迹起点，投掷准备
    robot.move_to(traj[0])?;

    let handle = robot.joint_impedance_async(
        &[600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0],
        &[50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0],
    )?;

    for (i, point) in traj.iter().enumerate() {
        if let MotionType::Joint(joint) = point {
            handle.set_target(Some(*joint));
            // 控制夹爪
            if gripper_traj[i] {
                // true: 夹紧
                gripper.grasp(0.03, 0.1, 1.0).unwrap();
            } else {
                // false: 松开
                gripper.move_gripper(0.08, 0.1).unwrap();
            }

            sleep(Duration::from_millis(1));
        }
    }

    handle.finish();

    Ok(())
}
