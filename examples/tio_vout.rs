use libjaka::types::{TioVout, TioVoutMode};

fn main() -> anyhow::Result<()> {
    let mut robot = libjaka::JakaMini2::new("10.5.5.100");
    robot.set_tio_vout(TioVout::Enable(TioVoutMode::V24V))?;
    let vout = robot.get_tio_vout()?;

    match vout {
        TioVout::Disable => println!("TIO Vout is disabled"),
        TioVout::Enable(TioVoutMode::V12V) => println!("TIO Vout is enabled: 12V"),
        TioVout::Enable(TioVoutMode::V24V) => println!("TIO Vout is enabled: 24V"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use libjaka::JakaMini2;
    use libjaka::types::{TioVout, TioVoutMode};
    use robot_behavior::behavior::*;

    #[test]
    fn enable_vout_12v() -> anyhow::Result<()> {
        let mut robot = JakaMini2::new("10.5.5.100");
        robot.set_tio_vout(TioVout::Enable(TioVoutMode::V12V))?;
        Ok(())
    }
}
