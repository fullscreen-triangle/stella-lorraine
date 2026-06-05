# Run from the repo root: .\crates\tempus\build_and_test.ps1
# Or from any directory: cargo test -p tempus

cargo build -p tempus
cargo test -p tempus -- --nocapture
