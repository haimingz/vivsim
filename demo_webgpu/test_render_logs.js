import { numpy as np, init } from "https://esm.sh/@jax-js/jax@0.1.11";
import { init_sim, update_fn, f_global } from "./main.js";

async function run() {
    await init();
    init_sim();
    const nx = 100, ny = 100;
    const f_next = update_fn(f_global);
    console.log("f_next computed");
}
run();
