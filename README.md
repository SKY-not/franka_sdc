# Readme

## 环境需求

- [Rust nightly](https://course.rs/first-try/installation.html)
  - 在 `windows` 中，需要先下载 [`C++ build tools`](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/)
  - 如果你之前安装过rust，那么你需要手动将rust的工具链切换到 `nightly` 版本，如果你没有安装过nightly需要安装
    - 安装 `rustup toolchain install nightly`
    - 运行 `rustup default nightly` 将工具链切换到 `nightly` 版本
- [Cmake](https://cmake.org/download/)
- rerun 运行 `cargo install --force rerun-cli@0.26.2`

## 说明
FrankaEmika 机械臂没有磁吸工装，所以`tio_vout.rs`中的代码无需使用