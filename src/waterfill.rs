//! Water-filling algorithm.
//!
//! Optimal power allocation for parallel Gaussian channels.

/// Solve the water-filling problem for parallel Gaussian channels.
///
/// Given noise powers n_i and total power P, finds optimal allocation p_i
/// that maximizes sum of (1/2) log2(1 + p_i / n_i).
///
/// Returns the power allocation vector.
pub fn water_filling(noise_powers: &[f64], total_power: f64) -> Vec<f64> {
    if noise_powers.is_empty() || total_power <= 0.0 {
        return vec![0.0; noise_powers.len()];
    }

    let n = noise_powers.len();
    let mut allocation = vec![0.0; n];

    // Sort channels by noise power (ascending) to find water level
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| noise_powers[a].partial_cmp(&noise_powers[b]).unwrap());

    // Iteratively find the water level
    let mut remaining_power = total_power;
    let mut active_channels = 0;

    for (rank, &idx) in indices.iter().enumerate() {
        active_channels = rank + 1;
        let water_level = noise_powers[idx] + remaining_power / active_channels as f64;

        // Check if all active channels can reach this water level
        let mut feasible = true;
        for &prev_idx in &indices[..rank] {
            if noise_powers[prev_idx] > water_level {
                feasible = false;
                break;
            }
        }

        if !feasible {
            active_channels = rank;
            break;
        }

        // Try to allocate
        let power_needed: f64 = indices[..=rank].iter()
            .map(|&i| (water_level - noise_powers[i]).max(0.0))
            .sum();

        if power_needed <= total_power {
            for &i in &indices[..=rank] {
                allocation[i] = (water_level - noise_powers[i]).max(0.0);
            }
            remaining_power = total_power - power_needed;
        } else {
            active_channels = rank;
            break;
        }
    }

    // Distribute remaining power equally among active channels
    if remaining_power > 0.0 && active_channels > 0 {
        let extra = remaining_power / active_channels as f64;
        for &i in &indices[..active_channels] {
            allocation[i] += extra;
        }
    }

    allocation
}

/// Compute the total capacity achieved by water-filling allocation.
pub fn water_filling_capacity(noise_powers: &[f64], allocation: &[f64]) -> f64 {
    noise_powers.iter().zip(allocation.iter())
        .map(|(&n, &p)| {
            if p > 0.0 {
                0.5 * (1.0 + p / n).log2()
            } else {
                0.0
            }
        })
        .sum()
}
