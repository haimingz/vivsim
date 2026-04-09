import { numpy as np } from "https://esm.sh/@jax-js/jax@0.1.11";

export function my_roll(ch, shift, axis) {
    const shape = ch.shape;
    const nx = shape[0];
    const ny = shape[1];

    if (axis === 0) {
        if (shift === 1) {
            const right = ch.ref.slice([nx-1, nx], [0, ny]);
            const left = ch.slice([0, nx-1], [0, ny]);
            return np.concatenate([right, left], 0);
        } else if (shift === -1) {
            const left = ch.ref.slice([0, 1], [0, ny]);
            const right = ch.slice([1, nx], [0, ny]);
            return np.concatenate([right, left], 0);
        }
    } else if (axis === 1) {
        if (shift === 1) {
            const top = ch.ref.slice([0, nx], [ny-1, ny]);
            const bottom = ch.slice([0, nx], [0, ny-1]);
            return np.concatenate([top, bottom], 1);
        } else if (shift === -1) {
            const bottom = ch.ref.slice([0, nx], [0, 1]);
            const top = ch.slice([0, nx], [1, ny]);
            return np.concatenate([top, bottom], 1);
        }
    }
    return ch;
}
