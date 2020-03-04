#![allow(non_snake_case)]

use super::smooth_line_search;
use crate::linop::{Adjoint, LinearOperator};
use ndarray::prelude::*;
use ndarray::NdFloat; // includes LinalgScalar and ScalarOperand
use ndarray_linalg::Scalar;
use num_traits::Float;

/// Optimized Gradient Method with a line search for L-Lipschitz Smooth Minimization
///
/// minimize a general objective function $`\sum_{j=1}^J f_j(B_j x)`$
/// where each $`f_j`$ has a $`L_{\nabla f_j}`$-Lipschitz smooth gradient.
/// see [\[DT18\]](#references).
///
/// Algorithm
/// ---------
/// ```math
/// \begin{aligned}
/// \theta_i &= \frac{1 + \sqrt{4 \theta_{i-1}^2 + 1}}{2} \\
/// y_i &= \left( 1 - \frac{1}{\theta_i} \right)x_{i-1} + \frac{1}{\theta_i}x_0 \\
/// d_i &= \left( 1 - \frac{1}{\theta_i} \right)\nabla f(x_{i-1}) + \frac{2}{\theta_i} \sum_{j=0}^{i-1} \theta_j \nabla f(x_j) \\
/// \alpha_i &\in \mathrm{arg}\!\min_{\alpha \in \mathbb{R}} f(y_i - \alpha d_i) \\
/// x_i &= y_i - \alpha_i d_i
/// \end{aligned}
/// ```
/// where $`\theta_0 := 1`$
///
/// Convergence
/// -----------
/// A worst-case converge of a smooth convex objective is given by
/// ```math
/// f(x_i) - f(x_*) \leq \frac{L \|x_0 - x_*\|_2^2}{2\theta_i^2}
/// ```
/// which has been shown to be optimal for this class of objectives.
///
/// Parameters
/// ----------
/// - __B:__         array of J blocks $`B_1,...,B_J`$  
/// - __grads:__     array of J functions for computing gradients of $`f_1,...,f_J`$  
/// - __L:__         array of J Lipschitz constants for those gradients  
/// - __x0:__        initial guess
/// - __maxiter:__   number of outer iterations  
/// - __ninner:__    number of inner iterations of line search  
/// - __callback:__  user-defined function to be evaluated with two arguments (x,itr).
///                   It is evaluated at (x0,0) and then after each iteration.
///                   If it returns True, the function terminates early.  
///
/// References
/// ----------
/// \[DT18\]: [ Drori Y, Taylor A
///             "Efficient First-order Methods for Convex Minimization: a
///             Constructive Approach", arxiv 1803.05676, (v2, Feb 2019) ](https://arxiv.org/abs/1803.05676)
pub fn ogm_els<'a, 'b, S, R, Q>(
    B: &'b [R],
    grads: &[impl (Fn(ArrayView<S, Ix1>) -> Array<S, Ix1>)],
    L: &[S],
    x0: ArrayView1<'a, S>,
    maxiter: usize,
    ninner: usize,
    mut callback: impl FnMut(ArrayView<S, Ix1>, usize) -> bool,
) -> Array<S, Ix1>
where
    S: NdFloat + Scalar,
    R: 'b + LinearOperator<Elem = S> + Adjoint<'b, Output = Q>,
    Q: LinearOperator<Elem = S>,
{
    let J = B.len();
    let grads = &grads[0..J];
    let L = &L[0..J];

    let mut thetai = S::one();

    let mut x = x0.to_owned();
    let x_dim = x.raw_dim();
    let Bx0: Vec<Array<S, Ix1>> = B.iter().map(|Bi| Bi.apply(&x0)).collect();
    let mut Bx: Vec<Array<S, Ix1>> = Bx0.iter().map(Array::to_owned).collect();

    let mut grad_sum: Array<S, Ix1> = Array::zeros(x.raw_dim());
    let grad = |Bx: &Vec<Array<S, Ix1>>| -> Array<S, Ix1> {
        (0..J)
            .map(|j| B[j].adj().apply(&grads[j](Bx[j].view())))
            .fold(Array::zeros(x_dim), |acc, g| acc + g)
    };
    let two = S::from(2.).unwrap();
    let four = S::from(4.).unwrap();
    let eight = S::from(8.).unwrap();

    if callback(x.view(), 0) {
        return x;
    };
    for iter in 1..=maxiter {
        // Compute conjugate direction
        let grad_new = grad(&Bx);
        grad_sum += &((&grad_new) * thetai);

        let constant = if iter < maxiter { four } else { eight };
        thetai = (S::one() + Float::sqrt(constant * thetai.powi(2) + S::one())) / two;

        let c0 = S::one() / thetai;
        let c1 = S::one() - c0;

        // y sequence (stored in x)
        x = (x) * c1 + (&x0) * c0; // y
        for (By_j, Bx0_j) in Bx.iter_mut().zip(Bx0.iter()) {
            By_j.mapv_inplace(|by| by * c1); // Bx * c1
            By_j.scaled_add(c0, Bx0_j); // + Bx0 * c0
        }

        let dir = (&grad_new) * (-c1) - (&grad_sum) * (two * c0);
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
            S::zero()
        };

        x.scaled_add(step, &dir); // x = y + step*dir
        for (Bx_j, Bdir_j) in Bx.iter_mut().zip(Bdir.iter()) {
            // Bx = By + step*Bdir
            Bx_j.scaled_add(step, Bdir_j);
        }
        if callback(x.view(), iter) {
            break;
        };
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test;

    #[test]
    fn ogm_els_simple_regression() {
        const NITER: usize = 150;
        let A = array![[10., 0.], [0., 20.]];
        let y = array![50., 100.];
        let x0 = array![19.0f32, 44.0f32];
        // Iterates check for regression
        let xk_ref: [[f32; 2]; 10 + 1] = [
            [19., 44.],
            [15.478901, 4.7648964],
            [12.704369, 4.534378],
            [9.738013, 4.5652933],
            [7.110388, 4.756363],
            [5.1809306, 4.9772825],
            [4.0755224, 5.112115],
            [3.6989412, 5.140752],
            [3.8336394, 5.108853],
            [4.238797, 5.0619874],
            [4.710328, 5.0216403],
        ];
        let mut finished = false;
        let x = ogm_els(
            &[A],
            &[(|x| &x - &y)],
            &[1.0],
            x0.view(),
            NITER,
            10,
            |x, iter| {
                assert!(iter <= NITER);
                println!("[{}, {}],", x[0], x[1]);
                if iter <= 10 {
                    let xk_star = xk_ref[iter];
                    assert_eq!(x[0], xk_star[0]);
                    assert_eq!(x[1], xk_star[1]);
                } else if iter == NITER {
                    finished = true;
                }
                false
            },
        );
        assert_abs_diff_eq!(x, array![5., 5.], epsilon = 0.000001);
        assert!(finished);
    }
}
