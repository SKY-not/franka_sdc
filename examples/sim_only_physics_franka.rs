use anyhow::Result;
use franka_rust::FrankaEmika;
use libjaka::JakaMini2;
use robot_behavior::behavior::*;
use rsbullet::{Mode, RsBullet};
use std::{f64::consts::FRAC_PI_4, thread::sleep, time::Duration};

fn main() -> Result<()> {
    let mut physics = RsBullet::new(Mode::Gui)?;
    physics
        .add_search_path("..\\asserts")?
        .set_gravity([0., 0., -9.81])?
        .set_step_time(Duration::from_secs_f64(1. / 240.))?;

    let mut robot_1 = physics
        .robot_builder::<JakaMini2>("robot_1")
        .base([0.0, 0.2, 0.0])
        .base_fixed(true)
        .load()?;
    let mut robot_2 = physics
        .robot_builder::<FrankaEmika>("robot_2")
        .base([0.0, -0.2, 0.0])
        .base_fixed(true)
        .load()?;

    for _ in 0..100_000 {
        physics.step()?;
    }
    robot_1
        .with_velocity(&[5.; 6])
        .with_acceleration(&[2.; 6])
        .move_joint(&[0.; 6])?;
    robot_2.move_joint(&[FRAC_PI_4; 7])?;

    loop {
        physics.step()?;
        sleep(Duration::from_secs_f64(0.01));
    }
}
