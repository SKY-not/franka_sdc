use std::{
    f64::consts::{FRAC_PI_2, PI},
    time::Duration,
};

use franka_rust::FrankaEmika;
use robot_behavior::{Pose, behavior::*};
use roplat_rerun::RerunHost;
use rsbullet::{DebugLineOptions, Mode, RsBullet, RsBulletRobot}; // 引入 nalgebra 处理数学计算

fn main() -> anyhow::Result<()> {
    let mut renderer = RerunHost::new("jaka_calibration")?;
    let mut physics_engine = RsBullet::new(rsbullet::Mode::Gui)?;
    // let mut physics_engine = RsBullet::new(rsbullet::Mode::Direct)?;

    physics_engine
        .add_search_path("./asserts")?
        .set_gravity([0., 0., -9.81])?
        .set_step_time(Duration::from_secs_f64(1. / 240.))?;
    renderer.add_search_path("./asserts")?;

    // 在物理引擎中加载机器人
    let mut robot = physics_engine
        .robot_builder::<FrankaEmika>("robot_1")
        .base([0.0, 0.0, 0.0])
        .base_fixed(true)
        .load()?;

    // 给每个关节添加坐标系可视化
    for link_index in &robot.joint_indices {
        physics_engine
            .client
            .add_user_debug_line(&DebugLineOptions {
                from: [0., 0., 0.],
                to: [0.2, 0., 0.],
                color: Some([1., 0., 0.]),
                line_width: 1.,
                life_time: 0.,
                parent_object_unique_id: Some(robot.body_id),
                parent_link_index: Some(*link_index),
                replace_item_unique_id: None,
            })?;

        physics_engine
            .client
            .add_user_debug_line(&DebugLineOptions {
                from: [0., 0., 0.],
                to: [0., 0.2, 0.],
                color: Some([0., 1., 0.]),
                line_width: 1.,
                life_time: 0.,
                parent_object_unique_id: Some(robot.body_id),
                parent_link_index: Some(*link_index),
                replace_item_unique_id: None,
            })?;

        physics_engine
            .client
            .add_user_debug_line(&DebugLineOptions {
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

    // 渲染器中加载一个相同的机器人用于可视化
    let robot_renderer = renderer
        .robot_builder::<FrankaEmika>("robot_2")
        .base([0.0, 0.0, 0.0])
        .base_fixed(true)
        .load()?;

    robot_renderer.attach_from(&mut robot)?;

    // let translation = na::Translation3::new(0.16, -0.20, -0.0);
    // let rotation = na::UnitQuaternion::from_euler_angles(PI, 0.0, 0.0);
    // let target_pose = na::Isometry3::from_parts(translation, rotation);

    // for _ in 0..10 {
    //     physics_engine.step()?;
    // }
    // let _ = robot.state()?;

    robot.move_traj_from_file("./traj/full_throw_trajectory.json")?;

    for _ in 0..10000 {
        physics_engine.step()?;
    }
    Ok(())
}
