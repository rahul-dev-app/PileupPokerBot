//! WebAssembly SIMD kernels for the MLP layers of the extension's value search (search.js).
//! Build: python chrome_extension/build_wasm.py (writes chrome_extension/extension/mlp.wasm).
#![no_std]
use core::arch::wasm32::*;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unreachable()
}

extern "C" {
    static __heap_base: u8;
}

/// Start of the free memory; search.js places its buffers and the weights after it.
#[no_mangle]
pub extern "C" fn heap_base() -> usize {
    core::ptr::addr_of!(__heap_base) as usize
}

const LANES: usize = 8; // f32x4 accumulators per pass: 32 outputs

/// out[r] = inp[r] @ w + b for rows r < n, with ReLU if `relu` != 0.
/// inp (n, dim), w (dim, m) row-major, b (m), out (n, m); m a multiple of 32; nz: scratch for dim indices.
/// Zero inputs are skipped (afterstate features and ReLU outputs are mostly zero).
#[no_mangle]
pub unsafe extern "C" fn dense(
    inp: *const f32, n: usize, dim: usize, w: *const f32, b: *const f32, m: usize, out: *mut f32, relu: u32,
    nz: *mut u32,
) {
    for r in 0..n {
        let x = inp.add(r * dim);
        let mut k = 0;
        for i in 0..dim {
            if *x.add(i) != 0.0 {
                *nz.add(k) = i as u32;
                k += 1;
            }
        }
        let o = out.add(r * m);
        let mut j = 0;
        while j < m {
            let mut acc = [f32x4_splat(0.0); LANES];
            for q in 0..LANES {
                acc[q] = v128_load(b.add(j + 4 * q) as *const v128);
            }
            for t in 0..k {
                let i = *nz.add(t) as usize;
                let xs = f32x4_splat(*x.add(i));
                let row = w.add(i * m + j);
                for q in 0..LANES {
                    acc[q] = f32x4_add(acc[q], f32x4_mul(xs, v128_load(row.add(4 * q) as *const v128)));
                }
            }
            for q in 0..LANES {
                let v = if relu != 0 { f32x4_pmax(acc[q], f32x4_splat(0.0)) } else { acc[q] };
                v128_store(o.add(j + 4 * q) as *mut v128, v);
            }
            j += 4 * LANES;
        }
    }
}

/// out[r] = inp[r] . w + b for rows r < n (the output layer); dim a multiple of 4.
#[no_mangle]
pub unsafe extern "C" fn dot(inp: *const f32, n: usize, dim: usize, w: *const f32, b: f32, out: *mut f32) {
    for r in 0..n {
        let x = inp.add(r * dim);
        let mut acc = f32x4_splat(0.0);
        let mut j = 0;
        while j < dim {
            acc = f32x4_add(acc, f32x4_mul(v128_load(x.add(j) as *const v128), v128_load(w.add(j) as *const v128)));
            j += 4;
        }
        *out.add(r) = f32x4_extract_lane::<0>(acc) + f32x4_extract_lane::<1>(acc) + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc) + b;
    }
}
