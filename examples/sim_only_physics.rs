use std::{f64::consts::FRAC_PI_2, time::Duration};

use franka_rust::FrankaEmika;
use robot_behavior::behavior::*;
use rsbullet::{Mode, RsBullet};

fn main() -> anyhow::Result<()> {
    let mut physics = RsBullet::new(Mode::Gui)?;

    physics
        // .add_search_path("./asserts")?
        .add_search_path("./asserts/franka_panda")? // 添加资源搜索路径
        .set_gravity([0., 0., -9.81])?
        .set_step_time(Duration::from_secs_f64(1. / 240.))?;

    // 仿真器基础配置
    let mut robot = physics
        .robot_builder::<FrankaEmika>("exam_robot")
        .base([0., 0., 0.])
        .load()?;

    // 给出你的机器人运动指令，当然你也可以在step循环里动态下发指令
    robot.move_joint(&[FRAC_PI_2; 7])?;
    // you can also use
    // ```rust
    //     robot.move_cartesian(&${YOUR_POSE})?;
    //     robot.move_traj_from_file("xxx")?;
    //     robot.move_joint_traj(${YOUR_JOINT_LIST})?;
    // ```
    // and more...

    loop {
        physics.step()?;
        std::thread::sleep(Duration::from_secs_f64(1. / 240.)); //暂停便于观察
    }
}
