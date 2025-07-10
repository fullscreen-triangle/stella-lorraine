/// Client modules for external package integration
/// 
/// **In Memory of Mrs. Stella-Lorraine Masunda**
/// 
/// This module provides client implementations for interfacing with
/// external packages in the Masunda Temporal Coordinate Navigator system.

pub mod kambuzuma_client;
pub mod kwasa_kwasa_client;
pub mod mzekezeke_client;
pub mod buhera_client;
pub mod consiousness_client;

// Re-export all client types for convenience
pub use kambuzuma_client::KambuzumaClient;
pub use kwasa_kwasa_client::KwasaKwasaClient;
pub use mzekezeke_client::MzekezekeClient;
pub use buhera_client::BuheraClient;
pub use consiousness_client::ConsciousnessClient;
