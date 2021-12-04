@echo off
setlocal
cd /D "%~dp0"

cargo run --package rafx-shader-processor -- ^
--glsl-path glsl/*.vert glsl/*.frag glsl/*.comp ^
--rs-mod-path src/shaders ^
--cooked-shaders-path cooked_shaders ^
--package-vk ^
--package-metal ^
--for-rafx-framework-crate && cargo fmt && cargo test --package rafx-plugins --features "legion"