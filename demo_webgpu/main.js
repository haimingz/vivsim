import { numpy as np, init, defaultDevice, jit } from "https://esm.sh/@jax-js/jax@0.1.11";
import { my_roll } from "./my_roll.js";

const RIGHT_DIRS = [1, 5, 8];
const LEFT_DIRS = [3, 7, 6];
const UP_DIRS = [2, 5, 6];
const DOWN_DIRS = [4, 7, 8];

function sum_channels(f, indices) {
    let sum = np.squeeze(f.ref.slice([indices[0], indices[0]+1]), 0);
    for (let i = 1; i < indices.length; i++) {
        const next = np.squeeze(f.ref.slice([indices[i], indices[i]+1]), 0);
        sum = sum.add(next.ref);
    }
    return sum;
}

function get_macroscopic(f) {
    const rho = f.ref.sum(0);

    const sum_right = sum_channels(f.ref, RIGHT_DIRS);
    const sum_left = sum_channels(f.ref, LEFT_DIRS);
    const sum_up = sum_channels(f.ref, UP_DIRS);
    const sum_down = sum_channels(f.ref, DOWN_DIRS);

    const ux_num = sum_right.sub(sum_left.ref);
    const uy_num = sum_up.sub(sum_down.ref);

    const ux = ux_num.div(rho.ref);
    const uy = uy_num.div(rho.ref);

    const u = np.stack([ux, uy], 0);
    return [rho, u];
}

function get_equilibrium(rho, u) {
    const ux = np.squeeze(u.ref.slice([0, 1]), 0);
    const uy = np.squeeze(u.ref.slice([1, 2]), 0);

    const ux2 = np.square(ux.ref);
    const uy2 = np.square(uy.ref);
    const u_sq = ux2.add(uy2.ref);

    const feq_channels = [];
    const w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36];
    const v = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]];

    for (let i = 0; i < 9; i++) {
        const wi = w[i];
        const vx = v[i][0];
        const vy = v[i][1];

        let uc;
        if (vx !== 0 && vy !== 0) {
            const vxx = ux.ref.mul(vx);
            const vyy = uy.ref.mul(vy);
            uc = vxx.add(vyy.ref);
        } else if (vx !== 0) {
            uc = ux.ref.mul(vx);
        } else if (vy !== 0) {
            uc = uy.ref.mul(vy);
        } else {
            uc = np.zerosLike(ux.ref);
        }

        const uc_sq = np.square(uc.ref);

        const term1 = uc.ref.mul(3);
        const term2 = uc_sq.mul(4.5);
        const term3 = u_sq.ref.mul(-1.5);

        let sum_terms = term1.add(term2.ref);
        sum_terms = sum_terms.add(term3.ref);
        sum_terms = sum_terms.add(1);

        const rhow = rho.ref.mul(wi);
        const feq_i = rhow.mul(sum_terms.ref);

        feq_channels.push(feq_i);
    }

    return np.stack(feq_channels, 0);
}

function collision_bgk(f, feq, omega) {
    const term1 = f.ref.mul(1 - omega);
    const term2 = feq.ref.mul(omega);
    const f_new = term1.add(term2.ref);
    return f_new;
}

function get_omega(nu) {
    return 1.0 / (3.0 * nu + 0.5);
}

function streaming(f) {
    const channels = [];
    const v = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]];
    for (let i = 0; i < 9; i++) {
        let ch = np.squeeze(f.ref.slice([i, i+1]), 0);
        const vx = v[i][0];
        const vy = v[i][1];

        if (vx !== 0) {
            const old_ch = ch;
            ch = my_roll(old_ch, vx, 0);
        }
        if (vy !== 0) {
            const old_ch = ch;
            ch = my_roll(old_ch, vy, 1);
        }
        channels.push(ch);
    }
    return np.stack(channels, 0);
}

function create_boundary_masks(nx, ny) {
    const mask_top_data = new Float32Array(nx * ny);
    const mask_bottom_data = new Float32Array(nx * ny);
    const mask_left_data = new Float32Array(nx * ny);
    const mask_right_data = new Float32Array(nx * ny);

    for (let x = 0; x < nx; x++) {
        for (let y = 0; y < ny; y++) {
            let idx = x * ny + y;
            if (y === ny - 1) mask_top_data[idx] = 1;
            if (y === 0) mask_bottom_data[idx] = 1;
            if (x === 0) mask_left_data[idx] = 1;
            if (x === nx - 1) mask_right_data[idx] = 1;
        }
    }

    const mask_top = np.array(mask_top_data).reshape([1, nx, ny]);
    const mask_bottom = np.array(mask_bottom_data).reshape([1, nx, ny]);
    const mask_left = np.array(mask_left_data).reshape([1, nx, ny]);
    const mask_right = np.array(mask_right_data).reshape([1, nx, ny]);

    return { mask_top, mask_bottom, mask_left, mask_right };
}

function bounce_back(f) {
    const opp_dirs = [0, 3, 4, 1, 2, 7, 8, 5, 6];
    const channels = [];
    for (let i = 0; i < 9; i++) {
        const ch = np.squeeze(f.ref.slice([opp_dirs[i], opp_dirs[i]+1]), 0);
        channels.push(ch);
    }
    return np.stack(channels, 0);
}

let top_mask, bottom_mask, left_mask, right_mask;
let mask_init = false;

function apply_boundaries(f, u0) {
    const nx = f.shape[1];
    const ny = f.shape[2];

    if (!mask_init) {
        const masks = create_boundary_masks(nx, ny);
        top_mask = masks.mask_top;
        bottom_mask = masks.mask_bottom;
        left_mask = masks.mask_left;
        right_mask = masks.mask_right;
        mask_init = true;
    }

    const f_bb = bounce_back(f.ref);

    const cond_left = np.equal(left_mask.ref, 1);
    let f_new = np.where(cond_left, f_bb.ref, f.ref);

    const cond_right = np.equal(right_mask.ref, 1);
    f_new = np.where(cond_right, f_bb.ref, f_new.ref);

    const cond_bottom = np.equal(bottom_mask.ref, 1);
    f_new = np.where(cond_bottom, f_bb.ref, f_new.ref);

    const rho_top = np.ones([nx, ny]);
    const u_top_x = np.full([nx, ny], u0);
    const u_top_y = np.zeros([nx, ny]);
    // u_top should be [2, nx, ny].
    const u_top_x_exp = np.expandDims(u_top_x, 0);
    const u_top_y_exp = np.expandDims(u_top_y, 0);
    const u_top = np.concatenate([u_top_x_exp.ref, u_top_y_exp.ref], 0);

    const feq_top = get_equilibrium(rho_top.ref, u_top.ref);

    const cond_top = np.equal(top_mask.ref, 1);
    f_new = np.where(cond_top, feq_top.ref, f_new.ref);

    return f_new;
}

export function update_fn_js(f, omega, u0) {
    const [rho, u] = get_macroscopic(f.ref);
    const feq = get_equilibrium(rho.ref, u.ref);
    let f_post_col = collision_bgk(f.ref, feq.ref, omega);
    let f_streamed = streaming(f_post_col.ref);
    let f_out = apply_boundaries(f_streamed.ref, u0);
    return f_out;
}

const nx = 100;
const ny = 100;

export let f_global = null;
export let update_fn = null;

let animationId = null;
let running = false;
const u0 = 0.3;
const Re_grid = 30.0;
const nu = u0 / Re_grid;
const omega = get_omega(nu);

export function init_sim() {
    const rho = np.ones([nx, ny]);
    const u = np.zeros([2, nx, ny]);
    f_global = get_equilibrium(rho.ref, u.ref);

    update_fn = (f) => update_fn_js(f, omega, u0);
}

function step_sim() {
    if (!running) return;

    for (let i = 0; i < 2; i++) {
        f_global = update_fn(f_global);
    }

    render();

    animationId = requestAnimationFrame(step_sim);
}

function render() {
    const [rho, u] = get_macroscopic(f_global.ref);

    const ux = np.squeeze(u.ref.slice([0, 1]), 0);
    const uy = np.squeeze(u.ref.slice([1, 2]), 0);

    const ux2 = np.square(ux.ref);
    const uy2 = np.square(uy.ref);
    const u_mag2 = ux2.add(uy2.ref);

    const u_mag = np.sqrt(u_mag2.ref);

    // js() might be synchronous in older versions or synchronous backend
    const promiseOrData = u_mag.js();
    Promise.resolve(promiseOrData).then(js_data => {
        let data = new Float32Array(nx * ny);
        for(let i=0; i<nx; i++) {
            for(let j=0; j<ny; j++) {
                data[i*ny + j] = js_data[i][j];
            }
        }
        const canvas = document.getElementById('sim-canvas');
        if(!canvas) return;

        // Setup internal resolution match
        canvas.width = nx;
        canvas.height = ny;

        const ctx = canvas.getContext('2d');
        const imgData = ctx.createImageData(nx, ny);

        let max_v = 0;
        for (let i = 0; i < data.length; i++) {
            if (data[i] > max_v) max_v = data[i];
        }

        if (max_v === 0) max_v = 1e-6;

        for (let x = 0; x < nx; x++) {
            for (let y = 0; y < ny; y++) {
                const val = data[x * ny + y];
                // using plasma-like coloring simple math
                const t = val / max_v;
                const r = Math.min(255, Math.floor(t * 255));
                const g = Math.min(255, Math.floor(t * 100)); // simple orangeish
                const b = Math.min(255, Math.floor((1 - t) * 255));

                const cy = ny - 1 - y;
                const idx = (cy * nx + x) * 4;

                imgData.data[idx] = r;         // R
                imgData.data[idx+1] = g;       // G
                imgData.data[idx+2] = b;       // B
                imgData.data[idx+3] = 255;     // A
            }
        }

        ctx.putImageData(imgData, 0, 0);
    });
}

async function main() {
    console.log("Initializing jax-js...");
    const devices = await init();
    console.log("Available devices:", devices);

    if (devices.includes("webgpu")) {
        defaultDevice("webgpu");
        console.log("Using WebGPU device.");
    } else {
        console.log("WebGPU not supported, falling back to Wasm.");
    }

    const startBtn = document.getElementById('btn-start');
    const stopBtn = document.getElementById('btn-stop');

    startBtn.addEventListener('click', () => {
        startBtn.disabled = true;
        stopBtn.disabled = false;
        running = true;
        if (!f_global) init_sim();
        step_sim();
    });

    stopBtn.addEventListener('click', () => {
        stopBtn.disabled = true;
        startBtn.disabled = false;
        running = false;
        if (animationId) cancelAnimationFrame(animationId);
    });
}

if (typeof document !== "undefined") {
    main().catch(console.error);
}
