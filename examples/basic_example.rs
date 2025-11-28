/*
basic_example 提供了控制实体机器人的快捷操作指令，
包括初始化、使能、禁用和关闭机器人等操作

*/
use libjaka::JakaMini2;
use robot_behavior::behavior::*;

fn main() -> anyhow::Result<()> {
    let mut robot = JakaMini2::new("10.5.5.100");

    // 完全等同于内置函数 robot._power_on()?;
    robot.init()?;
    robot.enable()?;

    // 此时你才可以发送运动指令

    robot.disable()?;
    // 完全等同于内置函数 robot._power_off()?;
    robot.shutdown()?;

    Ok(())
}

/// 一些测试函数可以使用快捷按钮，这使得开发一些临时使用的功能变得更加方便。你可以将其复制到任何部分，
/// 但是只有在 #[cfg(test)] 下才会被编译。
#[cfg(test)]
mod tests {
    #[test]
    fn power_on() -> anyhow::Result<()> {
        let mut robot = super::JakaMini2::new("10.5.5.100");
        robot._power_on()?;
        Ok(())
    }

    #[test]
    fn power_off() -> anyhow::Result<()> {
        let mut robot = super::JakaMini2::new("10.5.5.100");
        robot._power_off()?;
        Ok(())
    }

    #[test]
    fn enable() -> anyhow::Result<()> {
        let mut robot = super::JakaMini2::new("10.5.5.100");
        robot._enable()?;
        Ok(())
    }

    #[test]
    fn disable() -> anyhow::Result<()> {
        let mut robot = super::JakaMini2::new("10.5.5.100");
        robot._disable()?;
        Ok(())
    }
}
