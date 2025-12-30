use anyhow::Result;
use franka_rust::{FrankaEmika, FrankaGripper, types::robot_types::SetCollisionBehaviorData};
use robot_behavior::{MotionType, behavior::*};
use std::{fs::File, thread::sleep, time::Duration};

const PICK_JOINT: [f64; 7] = [0.5202528356353336, 0.5411086764837565, -0.4750576830733827, -2.292643344417538, 0.10296240303251479, 2.599058195988337, 0.7519023520034467];

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
    gripper.move_gripper(0.8, 0.2).unwrap();
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
    println!("Try picking");
    robot.move_joint(&PICK_JOINT)?;
    gripper.grasp(0.03, 0.1, 1.0).unwrap();
    println!("Have picked up the object");

    // 运动到轨迹起点，投掷准备
    robot.move_to(traj[0])?;
    println!("ready");
    robot.move_joint(&PICK_JOINT)?;
    println!("fuck");

    let handle = robot.joint_impedance_async(
        &[600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0],
        &[50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0],
    )?;

    for (i, point) in traj.iter().enumerate() {
        if let MotionType::Joint(joint) = point {
            handle.set_target(Some(*joint));
            // sleep(Duration::from_millis(1));
            // 控制夹爪
            // if gripper_traj[i] {
            //     // true: 夹紧
            //     gripper.grasp(0.03, 0.1, 1.0).unwrap();
            // } else {
            //     // false: 松开
            //     gripper.move_gripper(0.09, 0.2).unwrap();
            // }
            if !gripper_traj[i] {
                gripper.move_gripper(0.09, 0.2).unwrap();
                println!("Throw");
            }
            sleep(Duration::from_millis(1));
        }
    }

    handle.finish();

    Ok(())
}
