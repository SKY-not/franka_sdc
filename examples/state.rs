use libjaka::JakaMini2;
use robot_behavior::behavior::*;

fn main() -> anyhow::Result<()> {
    let mut robot = JakaMini2::new("10.5.5.100");
    let state = robot.state()?;
    println!("Robot State: {:?}", state);
    Ok(())
}

#[cfg(test)]
mod tests {
    use libjaka::JakaMini2;
    use robot_behavior::behavior::*;

    #[test]
    fn get_state() -> anyhow::Result<()> {
        let mut robot = JakaMini2::new("10.5.5.100");
        let q = robot.state()?.joint.unwrap();
        let pose = robot.state()?.pose_o_to_ee.unwrap().euler();
        println!("q:{q:?}\npose:{pose:?}");
        Ok(())
    }
}
