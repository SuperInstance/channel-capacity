# channel-capacity

Shannon channel capacity and information-theoretic algorithms in pure Rust with no external dependencies.

## Features

- **Discrete memoryless channels**: BSC, BEC, Z-channel, generic discrete channels
- **Mutual information**: I(X;Y), conditional entropy H(X|Y), joint entropy H(X,Y)
- **Channel capacity**: Maximum mutual information via Blahut-Arimoto
- **Water-filling**: Optimal power allocation for parallel Gaussian channels
- **Bounds**: Fano's inequality, channel coding theorem bounds, sphere-packing bounds

## Usage

```rust
use channel_capacity::*;

// Binary symmetric channel
let bsc = BSC::new(0.1);
println!("BSC(0.1) capacity = {}", bsc.capacity()); // ≈ 0.531

// Binary erasure channel
let bec = BEC::new(0.5);
println!("BEC(0.5) capacity = {}", bec.capacity()); // 0.5

// Mutual information
let px = vec![0.5, 0.5];
let channel = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
let mi = mutual_information(&px, &channel);

// Water-filling
let noise = vec![1.0, 2.0, 3.0];
let allocation = water_filling(&noise, 6.0);
```

## License

MIT
