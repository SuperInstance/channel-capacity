//! Information-theoretic bounds.
//!
//! Fano's inequality, channel coding theorem bounds.

/// Fano's inequality: H(X|Y) ≤ H(Pe) + Pe * log2(|X| - 1).
///
/// Gives an upper bound on the conditional entropy given error probability.
pub fn fano_inequality(error_prob: f64, alphabet_size: usize) -> f64 {
    if alphabet_size <= 1 {
        return 0.0;
    }
    let h_pe = if error_prob > 0.0 && error_prob < 1.0 {
        -(error_prob * error_prob.log2() + (1.0 - error_prob) * (1.0 - error_prob).log2())
    } else {
        0.0
    };
    h_pe + error_prob * ((alphabet_size - 1) as f64).log2()
}

/// Channel coding theorem bound: probability of error decreases exponentially
/// with block length for rates below capacity.
///
/// Returns an approximate upper bound on block error rate for rate R,
/// capacity C, and block length n.
pub fn channel_coding_theorem_bound(rate: f64, block_length: usize) -> f64 {
    if rate <= 0.0 {
        return 1.0;
    }
    // Simplified bound: P_e ≤ 2^(-n * E(R))
    // where E(R) is the random coding exponent (approximation)
    let gap = (1.0 - rate).max(0.01); // Gap to capacity (assuming capacity = 1)
    let exponent = gap * gap; // Simplified exponent
    2.0_f64.powf(-(block_length as f64 * exponent))
}

/// Sphere-packing bound (lower bound on error probability).
pub fn sphere_packing_bound(rate: f64, capacity: f64, block_length: usize) -> f64 {
    if rate >= capacity {
        return 1.0;
    }
    if rate <= 0.0 {
        return 0.0;
    }
    let gap = capacity - rate;
    2.0_f64.powf(-(block_length as f64 * gap * gap))
}

/// Converse bound: for rate R, achievable error probability is at least...
pub fn converse_bound(rate: f64, capacity: f64, block_length: usize) -> f64 {
    if rate > capacity {
        return 1.0; // Impossible to achieve rates above capacity
    }
    sphere_packing_bound(rate, capacity, block_length)
}
