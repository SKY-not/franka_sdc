use franka_rust::FrankaEmika;
use robot_behavior::behavior::*;

fn main() -> anyhow::Result<()> {
    let mut robot = FrankaEmika::new("10.5.5.100");
    let state = robot.state()?;
    println!("Robot State: {:?}", state);
    Ok(())
}

#[cfg(test)]
mod tests {
    use franka_rust::FrankaEmika;
    use robot_behavior::behavior::*;

    #[test]
    fn get_state() -> anyhow::Result<()> {
        let mut robot = FrankaEmika::new("10.5.5.100");
        let q = robot.state()?.joint.unwrap();
        let pose = robot.state()?.pose_o_to_ee.unwrap().euler();
        println!("q:{q:?}\npose:{pose:?}");
        Ok(())
    }
}
