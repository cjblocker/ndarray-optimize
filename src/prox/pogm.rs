//! Proximal Optimized Gradient Method

use ndarray::prelude::*;
use ndarray::NdFloat;

/// Proximal Optimized Gradient Method
pub fn pogm<S>(
    fgrad: impl Fn(ArrayView1<S>) -> Array1<S>,
    gprox: impl Fn(ArrayView1<S>, S) -> Array1<S>,
    x0: ArrayView1<S>,
    #[allow(non_snake_case)] Lf: S,
    muf: S,
    maxiter: usize,
    restart: bool,
    mut callback: impl FnMut(ArrayView1<S>, usize) -> bool,
) -> Array1<S>
where
    S: NdFloat,
{
    // sequences/ sets of iterates
    // do we need this many allocations?
    let mut x = x0.to_owned();
    let mut w: Array1<S> = x0.to_owned();
    let mut y = x0.to_owned();
    let mut z = x0.to_owned(); // dead init if not strong?

    let mut theta = S::one();
    let mut zeta = S::one(); // dead init if not strong?
    let step = -S::one() / Lf;

    let two = S::from(2.).unwrap();
    let four = S::from(4.).unwrap();
    let eight = S::from(8.).unwrap();

    // for the strong convex case
    let q = muf / Lf;
    let beta2 = (two + q - (q.powi(2) + eight * q).sqrt()).powi(2) / four / (S::one() - q);
    let gamma2 = (two + q - (q.powi(2) + eight * q).sqrt()) / two;
    let strong = muf > S::zero();

    if callback(x.view(), 0) {
        return x;
    }
    for iter in 1..=maxiter {
        // save last iter
        let w_old = w.to_owned();
        let x_old = x.to_owned();
        let theta_old = theta;

        // compute constants for this iteration
        let constant = if iter < maxiter { four } else { eight };
        theta = (S::one() + (constant * theta.powi(2) + S::one()).sqrt()) / two;
        let beta = if strong {
            beta2
        } else {
            (theta_old - S::one()) / theta
        }; // "nesterov" momentum
        let gamma = if strong { gamma2 } else { theta_old / theta }; // "OGM" momentum
        let c3 = beta / (Lf * zeta);
        zeta = (two * theta_old + theta - S::one()) / (Lf * theta);

        // perform update, needs optimization for allocation
        let grad = fgrad(x.view());
        let grad_step: Array1<S> = &grad * step;
        w = &x + &grad_step; // primary sequence
        let zmx = &z - &x;
        z = w.to_owned() + (&w - &w_old) * beta + (grad_step) * gamma + &zmx * c3;
        x = gprox(z.view(), zeta); // secondary sequence

        // restart momentum if needed
        if restart {
            let y_old = y.to_owned();
            let g: Array1<S> = grad + &(zmx / zeta);
            y = x_old + &g * step;
            if g.dot(&(&y - &y_old)) <= S::zero() {
                // could restart on cost instead
                // but we already have the grad info
                theta = S::one()
            }
            // paper includes sigma update here
            // but requires saving g
        }

        if callback(x.view(), iter) {
            break;
        }
    }
    x
}
