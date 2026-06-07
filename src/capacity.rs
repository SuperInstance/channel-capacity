//! Channel capacity computation.
//!
//! Computes the maximum mutual information over all input distributions.

use crate::channel::{BSC, BEC, ZChannel, DiscreteChannel};

/// Trait for channels that can compute their capacity.
pub trait Channel {
    /// Compute the channel capacity.
    fn capacity(&self) -> f64;
}

impl Channel for BSC {
    fn capacity(&self) -> f64 {
        BSC::capacity(self)
    }
}

impl Channel for BEC {
    fn capacity(&self) -> f64 {
        BEC::capacity(self)
    }
}

impl Channel for ZChannel {
    fn capacity(&self) -> f64 {
        ZChannel::capacity(self)
    }
}

impl Channel for DiscreteChannel {
    fn capacity(&self) -> f64 {
        DiscreteChannel::capacity(self)
    }
}

/// Compute channel capacity for any channel implementing the Channel trait.
pub fn channel_capacity<C: Channel>(channel: &C) -> f64 {
    channel.capacity()
}

/// Compute capacity of an AWGN channel with given SNR.
///
/// C = (1/2) log2(1 + SNR) bits per channel use.
pub fn awgn_capacity(snr_db: f64) -> f64 {
    let snr_linear = 10.0_f64.powf(snr_db / 10.0);
    0.5 * (1.0 + snr_linear).log2()
}

/// Compute the bandwidth-power tradeoff: C = B * log2(1 + P/(N0*B)).
pub fn shannon_capacity_bandwidth(bandwidth: f64, power: f64, noise_density: f64) -> f64 {
    if noise_density <= 0.0 || bandwidth <= 0.0 {
        return f64::INFINITY;
    }
    bandwidth * (1.0 + power / (noise_density * bandwidth)).log2()
}
