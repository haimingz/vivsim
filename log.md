**2024-09-17**

- Huge change. Add support for 3-DOF motion.
- Refactor the codebase for readability and consistency.
- Rewrite some doctrings.

**2024-09-14**

- I tested how much time each part of the VIV simulation takes (D=50, N_MARKER=4D, NX=40D, NY=20D, MDF=3). The code runs at 605it/s. If I turn off the IB part, it runs at about 695it/s. It means that the pure LBM calculation takes about 87% of the total time consumption while the rest 13% is related to the IB calculation. The IB part is quite efficient due to the refined IB region technique. The dominant of LBM computaion will be more pronouced when the calculation grid becomes larger.

**2024-09-13**

- Index the boundaries with index arrays instead of using indices. It gives a 8% speedup in the lid-driven cavity example (200x200 grid).

**2024-09-12**

- For the sake of consistency and convenient for cross-GPU parallization, I tried to rearrange the dimensions of arrays including f, feq, and u into a (cardinality, NX, NY) order instead of the original (NX, NY, cardinality) order. However, it turns out slowing down the code by about 10-20%. It could be because of the heavier indexing/slicing workload associated with the new data arrangement pattern. So this change is reverted until I can find more time look into the problem in the future.
- I tested the [XLA performance flags](https://jax.readthedocs.io/en/latest/gpu_performance_tips.html). It leds to a speed-up of about 10-15%.
