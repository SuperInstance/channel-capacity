//! Mutual information computation.
//!
//! Calculates I(X;Y) = Σ p(x,y) log(p(x,y) / (p(x)p(y))).

/// Compute mutual information I(X;Y) given input distribution and channel.
///
/// `px` is the input distribution, `py_given_x` is the channel matrix
/// where `py_given_x[x][y] = P(Y=y | X=x)`.
pub fn mutual_information(px: &[f64], py_given_x: &[Vec<f64>]) -> f64 {
    let n_in = px.len();
    let n_out = py_given_x[0].len();

    // Compute output distribution
    let mut qy = vec![0.0; n_out];
    for (x, row) in py_given_x.iter().enumerate().take(n_in) {
        for (y, &prob) in row.iter().enumerate().take(n_out) {
            qy[y] += px[x] * prob;
        }
    }

    // I(X;Y) = Σ_{x,y} p(x,y) log2(p(x,y) / (p(x)*p(y)))
    let mut mi = 0.0;
    for (x, row) in py_given_x.iter().enumerate().take(n_in) {
        for (y, &prob) in row.iter().enumerate().take(n_out) {
            let pxy = px[x] * prob;
            let pxqy = px[x] * qy[y];
            if pxy > 1e-15 && pxqy > 1e-15 {
                mi += pxy * (pxy / pxqy).log2();
            }
        }
    }

    mi.max(0.0)
}

/// Compute conditional entropy H(X|Y) = H(X) - I(X;Y).
pub fn conditional_entropy(px: &[f64], py_given_x: &[Vec<f64>]) -> f64 {
    let mi = mutual_information(px, py_given_x);

    // H(X)
    let hx: f64 = px.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum();

    (hx - mi).max(0.0)
}

/// Compute joint entropy H(X,Y).
pub fn joint_entropy(px: &[f64], py_given_x: &[Vec<f64>]) -> f64 {
    let n_in = px.len();
    let n_out = py_given_x[0].len();

    let mut h = 0.0;
    for (x, row) in py_given_x.iter().enumerate().take(n_in) {
        for (prob, _) in row.iter().take(n_out).map(|&p| (p, ())) {
            let pxy = px[x] * prob;
            if pxy > 1e-15 {
                h -= pxy * pxy.log2();
            }
        }
    }

    h
}
