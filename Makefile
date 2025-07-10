# Masunda Temporal Coordinate Navigator - Makefile
# In memory of Mrs. Stella-Lorraine Masunda

.PHONY: help build test clean install dev release check format lint security audit docker docs examples bench profile

# Default target
help: ## Show this help message
	@echo "🌟 Masunda Temporal Coordinate Navigator - Build System"
	@echo "   In memory of Mrs. Stella-Lorraine Masunda"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Build targets
build: ## Build the project in debug mode
	@echo "🔨 Building Masunda Navigator..."
	cargo build

release: ## Build the project in release mode with optimizations
	@echo "🚀 Building Masunda Navigator (release mode)..."
	cargo build --release

install: ## Install the binary to cargo bin directory
	@echo "📦 Installing Masunda Navigator..."
	cargo install --path .

# Development targets
dev: ## Start development environment
	@echo "🛠️  Starting development environment..."
	cargo watch -x "run"

# Testing targets
test: ## Run all tests
	@echo "🧪 Running tests..."
	cargo test

test-verbose: ## Run tests with verbose output
	@echo "🧪 Running tests (verbose)..."
	cargo test -- --nocapture

test-single: ## Run a single test (usage: make test-single TEST=test_name)
	@echo "🧪 Running single test: $(TEST)"
	cargo test $(TEST) -- --nocapture

integration-test: ## Run integration tests only
	@echo "🧪 Running integration tests..."
	cargo test --test integration_test

# Code quality targets
check: ## Run cargo check
	@echo "🔍 Checking code..."
	cargo check

format: ## Format code using rustfmt
	@echo "🎨 Formatting code..."
	cargo fmt

format-check: ## Check if code is formatted correctly
	@echo "🎨 Checking code formatting..."
	cargo fmt -- --check

lint: ## Run clippy linter
	@echo "🔍 Linting code..."
	cargo clippy -- -D warnings

lint-fix: ## Fix linting issues automatically
	@echo "🔧 Fixing linting issues..."
	cargo clippy --fix

# Documentation targets
docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	cargo doc --no-deps --open

docs-all: ## Generate documentation with dependencies
	@echo "📚 Generating documentation (with dependencies)..."
	cargo doc --open

# Benchmarking targets
bench: ## Run benchmarks
	@echo "📊 Running benchmarks..."
	cargo bench --features benchmarks

profile: ## Run profiling with perf
	@echo "📈 Profiling Masunda Navigator..."
	cargo build --release
	perf record --call-graph=dwarf ./target/release/masunda-navigator
	perf report

# Security targets
security: ## Run security audit
	@echo "🔒 Running security audit..."
	cargo audit

audit: ## Run dependency audit
	@echo "🔍 Running dependency audit..."
	cargo audit
	cargo deny check

# Docker targets
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t masunda-navigator .

docker-run: ## Run Docker container
	@echo "🐳 Running Docker container..."
	docker run -it --rm masunda-navigator

docker-compose-up: ## Start services with docker-compose
	@echo "🐳 Starting services with docker-compose..."
	docker-compose up -d

docker-compose-down: ## Stop services with docker-compose
	@echo "🐳 Stopping services with docker-compose..."
	docker-compose down

# Cleaning targets
clean: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	cargo clean
	rm -rf target/
	rm -rf logs/
	rm -rf tmp/
	rm -rf coverage/

clean-all: clean ## Clean all artifacts including logs and data
	@echo "🧹 Cleaning all artifacts..."
	rm -rf temporal_coordinates/
	rm -rf precision_logs/
	rm -rf memorial_validations/
	rm -rf system_metrics/

# Setup targets
setup: ## Set up development environment
	@echo "🔧 Setting up development environment..."
	rustup update
	rustup component add rustfmt clippy rust-analyzer
	cargo install cargo-watch cargo-audit cargo-deny
	@echo "✅ Development environment setup complete"

setup-pre-commit: ## Set up pre-commit hooks
	@echo "🔧 Setting up pre-commit hooks..."
	cp scripts/pre-commit .git/hooks/
	chmod +x .git/hooks/pre-commit

# Precision targets (Masunda-specific)
precision-test: ## Run precision validation tests
	@echo "🎯 Running precision validation tests..."
	cargo test --features high-precision -- precision

temporal-navigation: ## Test temporal coordinate navigation
	@echo "🧭 Testing temporal navigation..."
	cargo test test_temporal_coordinate_navigation -- --nocapture

memorial-validation: ## Test memorial framework validation
	@echo "🌟 Testing memorial framework validation..."
	cargo test test_memorial_framework -- --nocapture

predeterminism-proof: ## Generate predeterminism proof
	@echo "🔬 Generating predeterminism proof..."
	cargo test test_predeterminism_proof -- --nocapture

# Continuous Integration targets
ci: format-check lint test ## Run CI pipeline locally
	@echo "🤖 Running CI pipeline..."
	cargo audit
	cargo test --release

ci-coverage: ## Run tests with coverage
	@echo "📊 Running tests with coverage..."
	cargo tarpaulin --out html --output-dir coverage/

# Release targets
pre-release: clean format lint test security ## Prepare for release
	@echo "🚀 Preparing for release..."
	cargo build --release

release-tag: ## Create release tag
	@echo "🏷️  Creating release tag..."
	git tag -a v$(VERSION) -m "Release version $(VERSION)"
	git push origin v$(VERSION)

# Examples targets
examples: ## Run all examples
	@echo "📋 Running examples..."
	cargo run --example basic_navigation
	cargo run --example precision_demonstration
	cargo run --example memorial_validation

# Memorial targets (special)
memorial-message: ## Display memorial message
	@echo "🌟 Displaying memorial message..."
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "                 IN MEMORY OF MRS. STELLA-LORRAINE MASUNDA"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "This project is dedicated to proving that your death was not"
	@echo "random but occurred at predetermined coordinates within the"
	@echo "eternal oscillatory manifold that governs all reality."
	@echo ""
	@echo "Through unprecedented temporal precision, we honor your memory"
	@echo "and demonstrate the mathematical structure of predetermined fate."
	@echo "═══════════════════════════════════════════════════════════════"

# Default target when no target is specified
all: build test ## Build and test everything

# Environment variables
RUST_LOG ?= info
RUST_BACKTRACE ?= 1
VERSION ?= $(shell grep '^version' Cargo.toml | cut -d '"' -f 2)

# Export environment variables
export RUST_LOG
export RUST_BACKTRACE 