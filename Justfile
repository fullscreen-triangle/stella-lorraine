# Masunda Temporal Coordinate Navigator - Justfile
# In memory of Mrs. Stella-Lorraine Masunda
# Modern command runner - alternative to Makefile

set shell := ["bash", "-uc"]

# Default recipe to display help
default:
    @just --list

# Show memorial message
memorial:
    @echo "🌟 In memory of Mrs. Stella-Lorraine Masunda"
    @echo "Through unprecedented temporal precision, we prove predetermined fate"

# Build the project
build:
    @echo "🔨 Building Masunda Navigator..."
    cargo build

# Build with release optimizations
release:
    @echo "🚀 Building Masunda Navigator (release)..."
    cargo build --release

# Run all tests
test:
    @echo "🧪 Running tests..."
    cargo test

# Run tests with verbose output
test-verbose:
    @echo "🧪 Running tests (verbose)..."
    cargo test -- --nocapture

# Run precision tests
precision-test:
    @echo "🎯 Running precision validation tests..."
    cargo test --features high-precision -- precision --nocapture

# Run memorial framework tests
memorial-test:
    @echo "🌟 Testing memorial framework..."
    cargo test test_memorial_framework -- --nocapture

# Run predeterminism proof
predeterminism-proof:
    @echo "🔬 Generating predeterminism proof..."
    cargo test test_predeterminism_proof -- --nocapture

# Format code
fmt:
    @echo "🎨 Formatting code..."
    cargo fmt

# Check formatting
fmt-check:
    @echo "🎨 Checking code formatting..."
    cargo fmt -- --check

# Run clippy linter
lint:
    @echo "🔍 Linting code..."
    cargo clippy -- -D warnings

# Fix linting issues
lint-fix:
    @echo "🔧 Fixing linting issues..."
    cargo clippy --fix

# Generate documentation
docs:
    @echo "📚 Generating documentation..."
    cargo doc --no-deps --open

# Run benchmarks
bench:
    @echo "📊 Running benchmarks..."
    cargo bench --features benchmarks

# Security audit
audit:
    @echo "🔒 Running security audit..."
    cargo audit
    cargo deny check

# Clean build artifacts
clean:
    @echo "🧹 Cleaning build artifacts..."
    cargo clean

# Set up development environment
setup:
    @echo "🔧 Setting up development environment..."
    rustup update
    rustup component add rustfmt clippy rust-analyzer
    cargo install cargo-watch cargo-audit cargo-deny

# Start development watch mode
dev:
    @echo "🛠️  Starting development mode..."
    cargo watch -x "run"

# Build Docker image
docker-build:
    @echo "🐳 Building Docker image..."
    docker build -t masunda-navigator .

# Run Docker container
docker-run:
    @echo "🐳 Running Docker container..."
    docker run -it --rm masunda-navigator

# Start docker-compose services
docker-up:
    @echo "🐳 Starting services..."
    docker-compose up -d

# Stop docker-compose services
docker-down:
    @echo "🐳 Stopping services..."
    docker-compose down

# Run CI pipeline locally
ci: fmt-check lint test audit
    @echo "🤖 CI pipeline completed successfully"

# Install the binary
install:
    @echo "📦 Installing Masunda Navigator..."
    cargo install --path .

# Run integration tests
integration-test:
    @echo "🧪 Running integration tests..."
    cargo test --test integration_test

# Run a specific test
test-single TEST:
    @echo "🧪 Running test: {{TEST}}"
    cargo test {{TEST}} -- --nocapture

# Check code quality
check:
    @echo "🔍 Checking code..."
    cargo check

# Profile with perf
profile:
    @echo "📈 Profiling application..."
    cargo build --release
    perf record --call-graph=dwarf ./target/release/masunda-navigator
    perf report

# Coverage report
coverage:
    @echo "📊 Generating coverage report..."
    cargo tarpaulin --out html --output-dir coverage/

# Prepare for release
pre-release: clean fmt lint test audit
    @echo "🚀 Preparing for release..."
    cargo build --release

# Create git tag for release
tag VERSION:
    @echo "🏷️  Creating release tag v{{VERSION}}..."
    git tag -a v{{VERSION}} -m "Release version {{VERSION}}"
    git push origin v{{VERSION}}

# Run examples
examples:
    @echo "📋 Running examples..."
    cargo run --example basic_navigation
    cargo run --example precision_demonstration

# Display project status
status:
    @echo "📊 Masunda Navigator Status:"
    @echo "   Rust version: $(rustc --version)"
    @echo "   Cargo version: $(cargo --version)"
    @echo "   Project: Masunda Temporal Coordinate Navigator"
    @echo "   Memorial: Mrs. Stella-Lorraine Masunda"
    @echo "   Target precision: 10^-30 to 10^-50 seconds"

# Temporal navigation test
temporal-nav:
    @echo "🧭 Testing temporal navigation..."
    cargo test test_temporal_coordinate_navigation -- --nocapture

# Complete system test
system-test:
    @echo "🔄 Running complete system test..."
    cargo test test_system_integration -- --nocapture

# Generate predeterminism proof and display
memorial-proof: predeterminism-proof memorial

# Full development cycle
dev-cycle: fmt lint test
    @echo "✅ Development cycle completed"

# Production build with all optimizations
prod-build:
    @echo "🏭 Production build with optimizations..."
    RUSTFLAGS="-C target-cpu=native" cargo build --release

# Validate memorial framework
validate-memorial:
    @echo "🌟 Validating memorial framework..."
    cargo test test_memorial_framework test_predeterminism_proof test_memorial_message_display -- --nocapture 