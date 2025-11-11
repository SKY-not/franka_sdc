use libjaka::JakaMini2;
use robot_behavior::{MotionType, Pose, behavior::*};
use nalgebra as na;

fn main() -> anyhow::Result<()> {
    let mut robot = JakaMini2::new("10.5.5.100");

    robot.move_joint(&[0.;6])?;
    robot.move_to(MotionType::Joint([0.;6]))?;

    robot.move_cartesian(&Pose::Quat(na::Isometry3::identity()))?;
    robot.move_to(MotionType::Cartesian(Pose::Quat(na::Isometry3::identity())))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use libjaka::JakaMini2;
    use robot_behavior::behavior::*;

    #[test]
    fn move_to_default() -> RobotResult<()> {
        let mut robot = JakaMini2::new("10.5.5.100");
        robot.move_joint(&JakaMini2::JOINT_DEFAULT)?;
        Ok(())
    }
    
    #[test]
    fn move_to_packed() -> RobotResult<()> {
        let mut robot = JakaMini2::new("10.5.5.100");
        robot.move_joint(&JakaMini2::JOINT_PACKED)?;
        Ok(())
    }
}