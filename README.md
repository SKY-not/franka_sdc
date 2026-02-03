# Franka-SDC

这里是2025-秋季学习系统动力学与控制课程的先进实验，我们实现了一个关于操作 Franka Emika Panda 7自由度机械臂进行夹取并加速投掷的项目。

## 环境需求

- [Rust nightly](https://course.rs/first-try/installation.html)
  - 在 `windows` 中，需要先下载 [`C++ build tools`](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/)
  - 如果你之前安装过rust，那么你需要手动将rust的工具链切换到 `nightly` 版本，如果你没有安装过nightly需要安装
    - 安装 `rustup toolchain install nightly`
    - 运行 `rustup default nightly` 将工具链切换到 `nightly` 版本
- [Cmake](https://cmake.org/download/)
- rerun 运行 `cargo install --force rerun-cli@0.26.2`

## 致谢

感谢助教 Jizhou Yan 关于 Franka 机械臂 rust 接口的支持，帮助我们降低了项目的工作量
