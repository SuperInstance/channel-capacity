//! Discrete memoryless channels.
//!
//! Provides BSC, BEC, Z-channel, and generic discrete channel models.

/// Binary symmetric channel with crossover probability p.
#[derive(Debug, Clone)]
pub struct BSC {
    /// Crossover probability.
    pub p: f64,
}

impl BSC {
    /// Create a BSC with crossover probability `p`.
    pub fn new(p: f64) -> Self {
        assert!((0.0..=0.5).contains(&p), "p must be in [0, 0.5]");
        Self { p }
    }

    /// Compute channel capacity: C = 1 - H(p).
    pub fn capacity(&self) -> f64 {
        if self.p == 0.0 || self.p == 1.0 {
            return 1.0;
        }
        1.0 - binary_entropy(self.p)
    }

    /// Get transition matrix.
    pub fn transition_matrix(&self) -> Vec<Vec<f64>> {
        vec![
            vec![1.0 - self.p, self.p],
            vec![self.p, 1.0 - self.p],
        ]
    }
}

/// Binary erasure channel with erasure probability ε.
#[derive(Debug, Clone)]
pub struct BEC {
    /// Erasure probability.
    pub epsilon: f64,
}

impl BEC {
    /// Create a BEC with erasure probability `epsilon`.
    pub fn new(epsilon: f64) -> Self {
        assert!((0.0..=1.0).contains(&epsilon));
        Self { epsilon }
    }

    /// Compute channel capacity: C = 1 - ε.
    pub fn capacity(&self) -> f64 {
        1.0 - self.epsilon
    }

    /// Get transition matrix (3 outputs: 0, 1, erasure).
    pub fn transition_matrix(&self) -> Vec<Vec<f64>> {
        vec![
            vec![1.0 - self.epsilon, 0.0, self.epsilon],
            vec![0.0, 1.0 - self.epsilon, self.epsilon],
        ]
    }
}

/// Z-channel where 0→0 always, 1→0 with probability p.
#[derive(Debug, Clone)]
pub struct ZChannel {
    /// Flip probability (1→0).
    pub p: f64,
}

impl ZChannel {
    /// Create a Z-channel with flip probability `p`.
    pub fn new(p: f64) -> Self {
        assert!((0.0..=1.0).contains(&p));
        Self { p }
    }

    /// Compute channel capacity.
    pub fn capacity(&self) -> f64 {
        if self.p == 0.0 {
            return 1.0;
        }
        if self.p == 1.0 {
            return 0.0;
        }
        // Optimal input distribution found numerically
        // C = H((1-q)/(1-p^q)) - (1-p*q)*H(p) / (1-p^q)  (approximate)
        // Simplified: iterate to find optimal q
        let mut best_cap = 0.0;
        for i in 1..1000 {
            let q = i as f64 / 1000.0;
            let cap = self.capacity_for_input(q);
            if cap > best_cap {
                best_cap = cap;
            }
        }
        best_cap
    }

    fn capacity_for_input(&self, q: f64) -> f64 {
        // q = P(X=0)
        let p0_out = q + (1.0 - q) * self.p; // P(Y=0)
        let p1_out = (1.0 - q) * (1.0 - self.p); // P(Y=1)

        let h_y = if p0_out > 0.0 && p1_out > 0.0 {
            -(p0_out * p0_out.log2() + p1_out * p1_out.log2())
        } else {
            0.0
        };

        let h_y_given_x = (1.0 - q) * binary_entropy(self.p);

        h_y - h_y_given_x
    }
}

/// Generic discrete memoryless channel.
#[derive(Debug, Clone)]
pub struct DiscreteChannel {
    /// Transition probability matrix: channel[x][y] = P(Y=y | X=x).
    pub matrix: Vec<Vec<f64>>,
}

impl DiscreteChannel {
    /// Create a channel from its transition matrix.
    pub fn new(matrix: Vec<Vec<f64>>) -> Self {
        Self { matrix }
    }

    /// Compute capacity using Blahut-Arimoto.
    pub fn capacity(&self) -> f64 {
        let n = self.matrix.len();
        let mut px = vec![1.0 / n as f64; n];

        for _ in 0..200 {
            // Compute q(y)
            let n_out = self.matrix[0].len();
            let mut qy = vec![0.0; n_out];
            for (x, row) in self.matrix.iter().enumerate() {
                for (y, &prob) in row.iter().enumerate() {
                    qy[y] += px[x] * prob;
                }
            }

            // Update px
            let mut new_px = vec![0.0; n];
            for (x, row) in self.matrix.iter().enumerate() {
                let mut sum = 0.0;
                for (y, &prob) in row.iter().enumerate() {
                    if prob > 1e-15 && qy[y] > 1e-15 {
                        sum += prob * (prob / qy[y]).ln();
                    }
                }
                new_px[x] = px[x] * sum.exp();
            }

            let total: f64 = new_px.iter().sum();
            if total > 0.0 {
                for p in &mut new_px {
                    *p /= total;
                }
            }
            px = new_px;
        }

        // Compute mutual information at optimal input
        let n_out = self.matrix[0].len();
        let mut qy = vec![0.0; n_out];
        for (x, row) in self.matrix.iter().enumerate() {
            for (y, &prob) in row.iter().enumerate() {
                qy[y] += px[x] * prob;
            }
        }

        let mut cap = 0.0;
        for (x, row) in self.matrix.iter().enumerate() {
            for (y, &prob) in row.iter().enumerate() {
                let pxy = px[x] * prob;
                if pxy > 1e-15 && qy[y] > 1e-15 {
                    cap += pxy * (pxy / (px[x] * qy[y])).log2();
                }
            }
        }

        cap.max(0.0)
    }
}

/// Binary entropy function H(p) = -p*log2(p) - (1-p)*log2(1-p).
pub fn binary_entropy(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }
    -(p * p.log2() + (1.0 - p) * (1.0 - p).log2())
}
