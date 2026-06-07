//! # channel-capacity
//!
//! Shannon channel capacity and related algorithms in pure Rust.
//!
//! Provides discrete memoryless channels (BSC, BEC, Z-channel), mutual information
//! computation, capacity with cost constraints, and water-filling for parallel
//! Gaussian channels.

pub mod channel;
pub mod mutual_info;
pub mod capacity;
pub mod waterfill;
pub mod bounds;

pub use channel::{BSC, BEC, ZChannel, DiscreteChannel};
pub use mutual_info::mutual_information;
pub use capacity::channel_capacity;
pub use waterfill::water_filling;
pub use bounds::{fano_inequality, channel_coding_theorem_bound};

#[cfg(test)]
mod tests {
    use super::*;

    // === Channel tests ===
    #[test]
    fn test_bsc_capacity() {
        let bsc = BSC::new(0.1);
        let cap = bsc.capacity();
        // C = 1 - H(p) = 1 - H(0.1) ≈ 1 - 0.469 ≈ 0.531
        assert!((cap - 0.531).abs() < 0.02);
    }

    #[test]
    fn test_bsc_zero_crossover() {
        let bsc = BSC::new(0.0);
        assert!((bsc.capacity() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bsc_half_crossover() {
        let bsc = BSC::new(0.5);
        assert!(bsc.capacity().abs() < 1e-10);
    }

    #[test]
    fn test_bec_capacity() {
        let bec = BEC::new(0.5);
        // C = 1 - ε = 0.5
        assert!((bec.capacity() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_bec_zero_erasure() {
        let bec = BEC::new(0.0);
        assert!((bec.capacity() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_z_channel_capacity() {
        let zc = ZChannel::new(0.1);
        let cap = zc.capacity();
        assert!(cap > 0.0 && cap < 1.0);
    }

    #[test]
    fn test_discrete_channel_uniform() {
        let dc = DiscreteChannel::new(vec![
            vec![0.7, 0.2, 0.1],
            vec![0.1, 0.7, 0.2],
            vec![0.2, 0.1, 0.7],
        ]);
        let cap = dc.capacity();
        assert!(cap > 0.0 && cap < 2.0);
    }

    // === Mutual information tests ===
    #[test]
    fn test_mutual_info_bsc() {
        let px = vec![0.5, 0.5];
        let py_given_x = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
        let mi = mutual_information(&px, &py_given_x);
        // Should be close to BSC(0.1) capacity
        assert!(mi > 0.4 && mi < 0.6);
    }

    #[test]
    fn test_mutual_info_perfect_channel() {
        let px = vec![0.5, 0.5];
        let py_given_x = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mi = mutual_information(&px, &py_given_x);
        assert!((mi - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_info_noisy_channel() {
        let px = vec![0.5, 0.5];
        let py_given_x = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let mi = mutual_information(&px, &py_given_x);
        assert!(mi.abs() < 1e-10);
    }

    // === Capacity tests ===
    #[test]
    fn test_capacity_bsc() {
        let bsc = BSC::new(0.1);
        let cap = channel_capacity(&bsc);
        assert!((cap - bsc.capacity()).abs() < 1e-6);
    }

    #[test]
    fn test_capacity_bec() {
        let bec = BEC::new(0.3);
        let cap = channel_capacity(&bec);
        assert!((cap - 0.7).abs() < 1e-6);
    }

    // === Water-filling tests ===
    #[test]
    fn test_water_filling_basic() {
        let noise_powers = vec![1.0, 2.0, 3.0];
        let total_power = 6.0;
        let allocation = water_filling(&noise_powers, total_power);
        let total_alloc: f64 = allocation.iter().sum();
        assert!((total_alloc - total_power).abs() < 1e-6);
    }

    #[test]
    fn test_water_filling_equal_noise() {
        let noise_powers = vec![1.0, 1.0, 1.0];
        let total_power = 3.0;
        let allocation = water_filling(&noise_powers, total_power);
        // Should allocate equally
        for &a in &allocation {
            assert!((a - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_water_filling_low_noise_gets_more() {
        let noise_powers = vec![0.1, 10.0];
        let total_power = 5.0;
        let allocation = water_filling(&noise_powers, total_power);
        assert!(allocation[0] > allocation[1]);
    }

    #[test]
    fn test_water_filling_zero_power() {
        let noise_powers = vec![1.0, 2.0];
        let allocation = water_filling(&noise_powers, 0.0);
        assert!(allocation.iter().all(|&a| a.abs() < 1e-10));
    }

    // === Bounds tests ===
    #[test]
    fn test_fano_inequality() {
        // H(X|Y) ≤ H(Pe) + Pe*log2(|X|-1)
        let pe = 0.1;
        let alphabet_size = 2;
        let bound = fano_inequality(pe, alphabet_size);
        assert!(bound > 0.0 && bound < 1.0);
    }

    #[test]
    fn test_fano_zero_error() {
        let bound = fano_inequality(0.0, 4);
        assert!(bound.abs() < 1e-10);
    }

    #[test]
    fn test_channel_coding_bound() {
        let bound = channel_coding_theorem_bound(0.5, 100);
        // Block error rate should be positive and < 1
        assert!(bound > 0.0 && bound <= 1.0);
    }
}
