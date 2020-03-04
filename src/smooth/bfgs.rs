#![allow(non_snake_case)]

use super::smooth_line_search;
use crate::linop::{Adjoint, LinearOperator};
use ndarray::prelude::*;
use ndarray::NdFloat;
use ndarray_linalg::Scalar;
// use num_traits::Float;
use std::collections::VecDeque;

/// Limited-memory BFGS with an exact line search
///
/// Minimizes a smooth objective function of the form $`\sum_{j=1}^J f_j(B_j x)`$,
/// where each $`f_j`$ has a $`L_{\nabla f_j}`$-Lipschitz smooth gradient,
/// using the limited-memory Broyden Fletcher Goldfarb Shanno (BFGS) algorithm 
/// [\[N80\]](#references).
/// See also [Wikipedia](https://en.wikipedia.org/wiki/Limited-memory_BFGS).
///
/// Algorithm
/// ---------
/// ```math
/// \begin{aligned}
/// g_k &= \nabla f(x_k) \\
/// H_{k} &= (I - \rho_{k-1} s_{k-1} y_{k-1}^H)H_{k-1}(I -  \rho_{k-1}y_{k-1}s_{k-1}^H) + \rho_{k-1}s_{k-1} s_{k-1}^H \\
/// d_k &= -H_kg_k \\
/// \alpha_i &\in \mathrm{arg}\!\min_{\alpha \in \mathbb{R}} f(x_{i-1} - \alpha d_i) \\
/// x_i &= x_{i-1} + \alpha_i d_i \\
/// s_k &= x_{k} - x_{k-1} = \alpha_i d_i \\
/// y_k &= g_{k} - g_{k-1} \\
/// \rho_k &= 1/y_k^Hs_k 
/// \end{aligned}
/// ```
///
/// Parameters
/// ----------
/// - __B:__         array of J blocks $`B_1,...,B_J`$  
/// - __grads:__     array of J functions for computing gradients of $`f_1,...,f_J`$  
/// - __L:__         array of J Lipschitz constants for those gradients  
/// - __x0:__        initial guess  
/// - __H0:__        initial guess of hessian  
/// - __maxiter:__   number of outer iterations  
/// - __ninner:__    number of inner iterations of line search  
/// - __memory_len:__ number of corrections to remember for inverse hessian approximation.
///                     Original paper suggests 3 to 7 is sufficient.  
/// - __callback:__  user-defined function to be evaluated with two arguments (x,itr).
///                   It is evaluated at (x0,0) and then after each iteration.
///                   If it returns True, the function terminates early.  
/// References
/// ----------
/// \[N80\]: [ Nocedal, J,
///      "Updating Quasi-Newton Matrices With Limited Storage",
///         Mathematics of Computation, Vol 35, #151, July 1980, 773-782 ](https://courses.engr.illinois.edu/ece544na/fa2014/nocedal80.pdf)
pub fn lbfgs_els<'a, 'b, T, S, R, Q>(
    B: &'b [R],
    grads: &[impl (Fn(ArrayView<S, Ix1>) -> Array<S, Ix1>)],
    L: &[S],
    x0: ArrayViewMut1<'a, S>,
    H0: &T,
    maxiter: usize,
    ninner: usize,
    memory_len: usize,
    mut callback: impl FnMut(ArrayView<S, Ix1>, usize) -> bool,
) -> ArrayViewMut1<'a, S>
where
    S: NdFloat + Scalar,
    R: 'b + LinearOperator<Elem = S> + Adjoint<'b, Output = Q>,
    Q: LinearOperator<Elem = S>,
    T: LinearOperator<Elem = S>,
{
    let J = B.len();
    let grads = &grads[0..J];
    let L = &L[0..J];

    let mut memory: VecDeque<(S, Array1<S>, Array1<S>)> = VecDeque::with_capacity(memory_len);
    let mut gamma = S::from(1.).unwrap();
    let mut alphas = Vec::with_capacity(memory_len);

    let mut x = x0;
    let x_dim = x.raw_dim();
    let mut Bx = B.iter().map(|Bi| Bi.apply(&x)).collect();
    let mut grad_old = Array1::<S>::zeros(x.raw_dim()); // dead init
    let grad = |Bx: &Vec<Array<S, Ix1>>| -> Array<S, Ix1> {
        (0..J)
            .map(|j| B[j].adj().apply(&grads[j](Bx[j].view())))
            .fold(Array::zeros(x_dim), |acc, g| acc + g)
    };

    if callback(x.view(), 0) {
        return x;
    };
    for iter in 1..=maxiter {
        let grad_new = grad(&Bx);

        // Two-loop BFGS Update
        let dir = {
            let mut dir = &grad_new * S::from(-1.).unwrap();
            for (rho, s, y) in memory.iter().rev() {
                let alpha = s.dot(&dir) * (*rho);
                dir = dir - &(y * alpha);
                alphas.push(alpha);
            }
            dir = H0.apply_into(dir); // from BFGS?
            dir = dir * gamma;        // from L-BFGS?
            for (alpha, (rho, s, y)) in alphas.iter().rev().zip(memory.iter()) {
                let beta = y.dot(&dir) * (*rho);
                dir = dir + s * (*alpha - beta);
            }
            dir
        };
        let Bdir: Vec<Array<S, Ix1>> = B.iter().map(|Bi| Bi.apply(&dir)).collect();

        // Compute step size with line search
        let Lalph: S = (0..J).map(|j| L[j] * (&Bdir[j]).dot(&Bdir[j])).sum();
        let step = if Lalph > S::zero() {
            smooth_line_search(
                |alph| {
                    (0..J)
                        .map(|j| (&Bdir[j]).dot(&grads[j]((&Bdir[j] * alph + &Bx[j]).view())))
                        .sum()
                },
                Lalph,
                S::one(),
                S::zero(),
                ninner,
            )
        } else {
            // direction is zero vector
            return x;
        };

        let y = &grad_new - &grad_old;
        let s = &dir * step;
        let y_s = y.dot(&s);
        let rho = S::one() / y_s;
        gamma = y_s / y.dot(&y);
        if iter > memory_len {
            // forget
            memory.pop_front();
        }
        memory.push_back((rho, s, y));

        // Take step
        x.scaled_add(step, &dir);
        for (Bx_j, Bdir_j) in Bx.iter_mut().zip(Bdir.iter()) {
            Bx_j.scaled_add(step, Bdir_j);
        }
        if callback(x.view(), iter) {
            break;
        };

        grad_old = grad_new;
        alphas.clear();
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linop::Identity;
    use approx::assert_abs_diff_eq;
    use test;

    #[test]
    fn lbfgs_els_simple_regression() {
        const NITER: usize = 16;
        let A = array![[10., 0.], [0., 20.]];
        let y = array![50., 100.];
        let mut x0 = array![19.0f32, 44.0f32];
        // Iterates check for regression
        let xk_ref: [[f32; 2]; NITER + 1] = [
            [19., 44.],             // 0
            [15.478901, 4.7648964], // 1
            [5.245289, 5.683301],   // 2
            [5.061692, 5.6874204],  // 3
            [4.8207016, 5.0160923], // 4
            [4.8196197, 5.004053],  // 5
            [4.9957805, 4.9882436], // 6
            [4.998947, 4.988173],   // 7
            [5.0030937, 4.9997234], // 8
            [5.0031137, 4.999923],  // 9
            [5.0000715, 5.0001965], // 10
            [5.000009, 5.000197],   // 11
            [4.999941, 5.0000067],  // 12
            [4.9999404, 5.0000052], // 13
            [5.000001, 5.],         // 14
            [5.000001, 5.],         // 15
            [5.0000014, 5.000001],  // 16
        ];
        let mut finished = false;
        let x = lbfgs_els(
            &[A],
            &[(|x| &x - &y)],
            &[1.0],
            x0.view_mut(),
            &Identity::new(),
            NITER,
            10,
            5,
            |x, iter| {
                assert!(iter <= NITER);
                println!("[{}, {}],", x[0], x[1]);
                let xk_star = xk_ref[iter];
                assert_eq!(x[0], xk_star[0]);
                assert_eq!(x[1], xk_star[1]);
                if iter == NITER {
                    finished = true;
                }
                false
            },
        );
        assert_abs_diff_eq!(x, array![5., 5.], epsilon = 0.00001);
        assert_abs_diff_eq!(x0, array![5., 5.], epsilon = 0.00001);
        assert!(finished);
    }
}
