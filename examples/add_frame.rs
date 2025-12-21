use std::{f64::consts::FRAC_PI_2, time::Duration};

use libjaka::JakaMini2;
use robot_behavior::behavior::*;
use rsbullet::{DebugLineOptions, Mode, RsBullet, RsBulletRobot};

fn main() -> anyhow::Result<()> {
    let mut physics = RsBullet::new(Mode::Gui)?;

    physics
        .add_search_path("./asserts")?
        .set_gravity([0., 0., -10.])?
        .set_step_time(Duration::from_secs_f64(1. / 240.))?;

    // 仿真器基础配置
    let mut robot: RsBulletRobot<JakaMini2> = physics
        .robot_builder::<JakaMini2>("exam_robot")
        .base([0., 0., 0.])
        .load()?;

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

    // 给出你的机器人运动指令，当然你也可以在step循环里动态下发指令
    robot.move_joint(&[FRAC_PI_2; _])?;
    // you can also use
    // ```rust
    //     robot.move_cartesian(&${YOUR_POSE})?;
    //     robot.move_traj_from_file("xxx")?;
    //     robot.move_joint_traj(${YOUR_JOINT_LIST})?;
    // ```
    // and more...

    loop {
        physics.step()?;
    }
}