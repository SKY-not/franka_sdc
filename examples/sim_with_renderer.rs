use std::f64::consts::FRAC_PI_2;

use franka_rust::FrankaEmika;
use robot_behavior::behavior::*;
use roplat_rerun::RerunHost;
use rsbullet::{Mode, RsBullet};

fn main() -> anyhow::Result<()> {
    let mut physics = RsBullet::new(Mode::Gui)?;
    let mut renderer = RerunHost::new("mini_exam")?;

    let mut robot = physics
        .robot_builder::<FrankaEmika>("exam_robot")
        .base([0., 0., 0.])
        .load()?;

    let robot_render = renderer
        .robot_builder::<FrankaEmika>("exam_robot")
        .base([0., 0., 0.])
        .load()?;

    robot_render.attach_from(&mut robot)?;
    robot.move_joint(&[FRAC_PI_2; 7])?;

    loop {
        physics.step()?;
    }
}
