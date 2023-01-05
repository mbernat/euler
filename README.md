## Eulerian fluid simulation

Install Rust and use `cargo run` to run on GPU or `cargo run --release` to run on CPU.

GPU implementation is currently much slower due to using the Jacobi method for pressure projection, which is much slower than Gauss--Seidel with over-relaxation on the CPU.