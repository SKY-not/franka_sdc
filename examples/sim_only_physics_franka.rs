use anyhow::Result;
use franka_rust::FrankaEmika;
use robot_behavior::behavior::*;
use rsbullet::{DebugLineOptions, Mode, RsBullet, RsBulletRobot};
use std::{f64::consts::FRAC_PI_4, thread::sleep, time::Duration};

fn main() -> Result<()> {
    let mut physics = RsBullet::new(Mode::Gui)?;
    physics
        .add_search_path("./asserts")?
        .set_gravity([0., 0., -9.81])?
        // .set_step_time(Duration::from_secs_f64(0.1))?;
        .set_step_time(Duration::from_secs_f64(1. / 240.))?;

    let mut robot = physics
        .robot_builder::<FrankaEmika>("robot")
        .base([0.0, 0.0, 0.0])
        .base_fixed(true)
        .load()?;

    // 给每个关节添加坐标系可视化
    for link_index in &robot.joint_indices {
        physics.client.add_user_debug_line(&DebugLineOptions {
            from: [0., 0., 0.],
            to: [0.2, 0., 0.],
            color: Some([1., 0., 0.]),
            line_width: 1.,
            life_time: 0.,
            parent_object_unique_id: Some(robot.body_id),
            parent_link_index: Some(*link_index),
            replace_item_unique_id: None,
        })?;

        physics.client.add_user_debug_line(&DebugLineOptions {
            from: [0., 0., 0.],
            to: [0., 0.2, 0.],
            color: Some([0., 1., 0.]),
            line_width: 1.,
            life_time: 0.,
            parent_object_unique_id: Some(robot.body_id),
            parent_link_index: Some(*link_index),
            replace_item_unique_id: None,
        })?;

        physics.client.add_user_debug_line(&DebugLineOptions {
            from: [0., 0., 0.],
            to: [0., 0., 0.2],
            color: Some([0., 0., 1.]),
            line_width: 1.,
            life_time: 0.,
            parent_object_unique_id: Some(robot.body_id),
            parent_link_index: Some(*link_index),
            replace_item_unique_id: None,
        })?;
    }

    robot.move_joint(&FrankaEmika::JOINT_DEFAULT)?;
    for _ in 0..10000 {
        physics.step()?;
    }
    // 根据json文件运动
    robot.move_traj_from_file("./traj/full_throw_trajectory.json")?;
    // robot.move_traj_from_file("./traj/ik_result.json")?;
    loop {
        physics.step()?;
        sleep(Duration::from_secs_f64(1. / 240.));
    }
    // Ok(())
}
