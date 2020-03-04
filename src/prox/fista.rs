//! Fast Iterative Shrinking/Thresholding Algorithm

use ndarray::prelude::*;
use ndarray::NdFloat;

/// Fast Iterative Shrinking/Thresholding Algorithm
pub fn fista<S>(
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
    let mut y = x0.to_owned();

    let mut theta = S::one();
    let step = S::one() / Lf;

    let two = S::from(2.).unwrap();
    let four = S::from(4.).unwrap();
    let eight = S::from(8.).unwrap();

    // for the strong convex case
    let q = muf / Lf;
    let beta2 = (two + q - (q.powi(2) + eight * q).sqrt()).powi(2) / four / (S::one() - q);
    let strong = muf > S::zero();

    if callback(x.view(), 0) {
        return x;
    }
    for iter in 1..=maxiter {
        // save last iter
        let y_old = y.to_owned();
        let x_old = x.to_owned();
        let theta_old = theta;

        // compute constants for this iteration
        theta = (S::one() + (four * theta.powi(2) + S::one()).sqrt()) / two;
        let beta = if strong {
            beta2
        } else {
            (theta_old - S::one()) / theta
        }; // "nesterov" momentum

        // perform update, needs optimization for allocation
        let grad = fgrad(y.view());
        x = gprox((y - &grad*step).view(), step);
        y = &x + &((&x - &x_old)*beta);

        // restart momentum if needed
        if restart {
            let g: Array1<S> = grad - (&x - &y_old)*Lf;
            if g.dot(&(&x - &x_old)) <= S::zero() {
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
