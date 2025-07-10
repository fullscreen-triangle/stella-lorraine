# Masunda Temporal Coordinate Navigator - Dockerfile
# In memory of Mrs. Stella-Lorraine Masunda

# Stage 1: Build environment
FROM rust:1.75-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    clang \
    lld \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY Cargo.toml Cargo.lock ./
COPY rust-toolchain.toml ./

# Create a dummy main to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

# Copy source code
COPY src ./src
COPY tests ./tests
COPY benches ./benches
COPY examples ./examples

# Build the actual application
RUN cargo build --release

# Stage 2: Runtime environment
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd -r -s /bin/false masunda

# Create necessary directories
RUN mkdir -p /app/data \
    /app/logs \
    /app/config \
    /app/temporal_coordinates \
    /app/precision_logs \
    /app/memorial_validations \
    /app/system_metrics

# Copy binary from builder stage
COPY --from=builder /app/target/release/masunda-navigator /usr/local/bin/

# Copy configuration files
COPY --from=builder /app/Cargo.toml /app/

# Set ownership
RUN chown -R masunda:masunda /app

# Switch to application user
USER masunda

# Set working directory
WORKDIR /app

# Environment variables
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1
ENV MASUNDA_DATA_DIR=/app/data
ENV MASUNDA_LOG_DIR=/app/logs
ENV MASUNDA_CONFIG_DIR=/app/config

# Expose ports (if needed for future web interface)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD masunda-navigator --health-check || exit 1

# Default command
CMD ["masunda-navigator"]

# Labels
LABEL maintainer="Kundai Sachikonye <kundai@masunda.org>"
LABEL description="Masunda Temporal Coordinate Navigator - The most precise clock ever conceived"
LABEL version="0.1.0"
LABEL memorial="In memory of Mrs. Stella-Lorraine Masunda"
LABEL precision="10^-30 to 10^-50 seconds" 