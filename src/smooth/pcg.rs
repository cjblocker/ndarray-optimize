//! Private Module

#![allow(non_snake_case)]

use super::smooth_line_search;
use crate::linop::{Adjoint, LinearOperator};
use ndarray::prelude::*;
use ndarray::NdFloat; // includes LinalgScalar and ScalarOperand
use ndarray_linalg::Scalar;

/// Choice of update of $`\gamma`$ for conjugate gradient
///
/// All methods are equivalent on a quadratic problem, but
/// represent different relaxations for non-quadratic problems
#[derive(Debug)]
pub enum Gamma {
    /// ```math
    /// \gamma_i = \frac{ \langle g_i - g_{i-1}, Pg_i \rangle}{\langle g_i - g_{i-1}, d_{i-1} \rangle }
    /// ```
    HestenesStiefel,

    /// ```math
    /// \gamma_i = \frac{ \langle g_i - g_{i-1}, Pg_i \rangle}{\langle g_{i-1}, Pg_{i-1} \rangle }
    /// ```
    PolakRibiere,

    /// ```math
    /// \gamma_i = \frac{ \langle g_i, Pg_i \rangle}{\langle g_i - g_{i-1}, d_{i-1} \rangle }
    /// ```
    DaiYuan,

    /// ```math
    /// \gamma_i = \frac{ \langle g_i, Pg_i \rangle}{\langle g_{i-1}, Pg_{i-1} \rangle }
    /// ```
    FletcherReeves,

    /// ```math
    /// \begin{aligned}
    /// \zeta_i &= 2 \frac{ \langle g_i - g_{i-1}, P(g_i - g_{i-1}) \rangle}{\langle g_i - g_{i-1}, d_{i-1} \rangle } \\
    /// \gamma_i &= \frac{ \langle P(g_i - g_{i-1}) - \zeta_i d_{i-1}, g_i \rangle}{\langle g_i - g_{i-1}, d_{i-1} \rangle } \\
    /// \end{aligned}
    /// ```
    HagerZhang,
}

/// Preconditioned Non-linear Conjugate Gradient for L-Lipschitz Smooth Minimization
///
/// Algorithm
/// ---------
/// ```math
/// \begin{aligned}
/// g_i &= \nabla f(x_{i-1}) \\
/// \gamma_i &= \mathrm{gamma\_update}(g_i, g_{i-1}, d_{i-1}, P)  \\
/// d_i &= -Pg_i + \gamma_i d_{i-1} \\
/// \alpha_i &\in \mathrm{arg}\!\min_{\alpha \in \mathbb{R}} f(y_i - \alpha d_i) \\
/// x_i &= x_{i-1} + \alpha_i d_i
/// \end{aligned}
/// ```
/// where the choice of $`\gamma`$ update can be:
/// - [Hestenes-Stiefel ](enum.Gamma.html#variant.HestenesStiefel)
/// - [Polak-Ribiere ](enum.Gamma.html#variant.PolakRibiere)
/// - [Dai-Yuan ](enum.Gamma.html#variant.DaiYuan)
/// - [Fletcher-Reeves ](enum.Gamma.html#variant.FletcherReeves)
/// - [Hager-Zhang ](enum.Gamma.html#variant.HagerZhang)
///
/// Parameters
/// ----------
/// - __grad:__      function that computes gradient g(x) of a convex cost function  
/// - __L:__         Lipschitz constant of cost function gradient  
/// - __x0:__        initial guess  
/// - __P:__         preconditioner  
/// - __maxiter:__   number of outer NCG iterations  
/// - __ninner:__    number of inner iterations of GD for line search  
/// - __callback:__  User-defined function to be evaluated with two arguments (x,iter).
///                   It is evaluated at (x0,0) and then after each iteration.
///                   If it returns True, the function terminates early.
pub fn ncg<'a, T, S>(
    grad: impl (Fn(ArrayView<S, Ix1>) -> Array<S, Ix1>),
    L: S,
    x0: ArrayViewMut1<'a, S>,
    P: &T,
    maxiter: usize,
    ninner: usize,
    mut callback: impl FnMut(ArrayView<S, Ix1>, usize) -> bool,
) -> ArrayViewMut1<'a, S>
where
    S: NdFloat + Scalar,
    T: LinearOperator<Elem = S>,
{
    // loop variables (pass from one iter to next)
    let mut x = x0;
    let mut dir_old = Array::zeros(x.raw_dim()); // dead init
    let mut ngrad_old = Array::zeros(x.raw_dim()); // dead init

    if callback(x.view(), 0) {
        return x;
    };
    for iter in 1..=maxiter {
        // Compute conjugate direction
        let ngrad_new = -grad(x.view());
        let nPgrad_new = P.apply(&ngrad_new);
        let dir = if iter > 1 {
            let grad_Pnorm = (&ngrad_new).dot(&nPgrad_new);
            if grad_Pnorm == S::zero() {
                return x;
            }
            let gamma = grad_Pnorm / (&ngrad_old - &ngrad_new).dot(&dir_old);
            (&dir_old) * gamma + &nPgrad_new
        } else {
            nPgrad_new
        };

        // Compute step size with line search
        let Lalph = L * (&dir).dot(&dir);
        let step = if Lalph > S::zero() {
            smooth_line_search(
                |alph| (&dir).dot(&grad((&dir * alph + &x).view())),
                Lalph,
                S::one(),
                S::zero(),
                ninner,
            )
        } else {
            S::zero()
        };

        x.scaled_add(step, &dir);
        if callback(x.view(), iter) {
            break;
        };

        // update loop state variables
        ngrad_old = ngrad_new;
        dir_old = dir;
    }
    x
}

/// Preconditioned Non-linear Conjugate Gradient for L-Lipschitz Smooth Minimization
///
/// minimize a general objective function $`\sum_{j=1}^J f_j(B_j x)`$
/// where each $`f_j`$ has a $`L_{\nabla f_j}`$-Lipschitz smooth gradient.
///
/// Conjugate-gradient methods modify the search direction to ensure they are
/// conjugate (approximately so for non-quadratics).
///
/// Algorithm
/// ---------
/// ```math
/// \begin{aligned}
/// g_i &= \nabla f(x_{i-1}) \\
/// \gamma_i &= \mathrm{gamma\_update}(g_i, g_{i-1}, d_{i-1}, P)  \\
/// d_i &= -Pg_i + \gamma_i d_{i-1} \\
/// \alpha_i &\in \mathrm{arg}\!\min_{\alpha \in \mathbb{R}} f(x_{i-1} - \alpha d_i) \\
/// x_i &= x_{i-1} + \alpha_i d_i
/// \end{aligned}
/// ```
/// where the choice of $`\gamma`$ update can be:
/// - [Hestenes-Stiefel ](enum.Gamma.html#variant.HestenesStiefel)
/// - [Polak-Ribiere ](enum.Gamma.html#variant.PolakRibiere)
/// - [Dai-Yuan ](enum.Gamma.html#variant.DaiYuan)
/// - [Fletcher-Reeves ](enum.Gamma.html#variant.FletcherReeves)
/// - [Hager-Zhang ](enum.Gamma.html#variant.HagerZhang)
///
/// Parameters
/// ----------
/// - __B:__         array of J blocks $`B_1,...,B_J`$  
/// - __grads:__     array of J functions for computing gradients of $`f_1,...,f_J`$  
/// - __L:__         array of J Lipschitz constants for those gradients  
/// - __x0:__        initial guess  
/// - __P:__         preconditioner  
/// - __maxiter:__   number of outer iterations  
/// - __ninner:__    number of inner iterations of line search  
/// - __callback:__  user-defined function to be evaluated with two arguments (x,itr).
///                   It is evaluated at (x0,0) and then after each iteration.
///                   If it returns True, the function terminates early.  
#[allow(clippy::too_many_arguments)]
pub fn ncg_els<'a, 'b, S, T, R, Q>(
    B: &'b [R],
    grads: &[impl (Fn(ArrayView<S, Ix1>) -> Array<S, Ix1>)],
    L: &[S],
    x0: ArrayViewMut1<'a, S>,
    P: &T,
    maxiter: usize,
    ninner: usize,
    mut callback: impl FnMut(ArrayView<S, Ix1>, usize) -> bool,
) -> ArrayViewMut1<'a, S>
where
    S: NdFloat + Scalar,
    T: LinearOperator<Elem = S>,
    R: 'b + LinearOperator<Elem = S> + Adjoint<'b, Output = Q>,
    Q: LinearOperator<Elem = S>,
{
    let J = B.len();
    let grads = &grads[0..J];
    let L = &L[0..J];

    let mut x = x0;
    let x_dim = x.raw_dim();
    let mut Bx = B.iter().map(|Bi| Bi.apply(&x)).collect();
    let mut dir_old = Array::zeros(x.raw_dim()); // dead init
    let mut ngrad_old = Array::zeros(x.raw_dim()); // dead init
    let grad = |Bx: &Vec<Array<S, Ix1>>| -> Array<S, Ix1> {
        (0..J)
            .map(|j| B[j].adj().apply(&grads[j](Bx[j].view())))
            .fold(Array::zeros(x_dim), |acc, g| acc + g)
    };

    if callback(x.view(), 0) {
        return x;
    };
    for iter in 1..=maxiter {
        // Compute conjugate direction
        let ngrad_new = -grad(&Bx);
        let nPgrad_new = P.apply(&ngrad_new);
        let dir = if iter > 1 {
            let grad_Pnorm = (&ngrad_new).dot(&nPgrad_new);
            if grad_Pnorm == S::zero() {
                return x;
            }
            let gamma = grad_Pnorm / (&ngrad_old - &ngrad_new).dot(&dir_old);
            (&dir_old) * gamma + &nPgrad_new
        } else {
            nPgrad_new
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
            S::zero()
        };

        // Take step
        x.scaled_add(step, &dir);
        for (Bx_j, Bdir_j) in Bx.iter_mut().zip(Bdir.iter()) {
            Bx_j.scaled_add(step, Bdir_j);
        }
        if callback(x.view(), iter) {
            break;
        };

        // update loop variables
        ngrad_old = ngrad_new;
        dir_old = dir;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linop::Identity;
    use approx::assert_abs_diff_eq;
    use test;

    use ndarray::Array;
    use ndarray_linalg::norm::Norm;
    use ndarray_linalg::svd::SVD;
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;

    #[test]
    fn ncg_simple_regression() {
        const NITER: usize = 13;
        let A = array![[10., 0.], [0., 20.]];
        let y = array![50., 100.];
        let mut x0 = array![19.0f32, 44.0f32];
        // This Rust code actually converged
        // faster than the reference Numpy
        // so this is just what the current
        // iterates were to test for regression.
        let xk_ref: [[f32; 2]; NITER + 1] = [
            [19.0, 44.0],           // 0
            [15.478901, 4.7648964], // 1
            [5.5871410, 4.9868274], // 2 (didn't finish line search)
            [5.0425186, 5.0336366], // 3
            [5.0207340, 4.9969244], // 4
            [5.0048304, 5.0029430], // 5
            [5.0009240, 4.9997215], // 6
            [5.0004854, 5.0001300], // 7
            [5.0000570, 4.9999690], // 8
            [5.0000296, 5.0000057], // 9
            [5.0000060, 4.9999960], // 10
            [5.0000014, 5.0000005], // 11
            [5.0000005, 4.9999995], // 12
            [5., 5.],               // 13
        ];
        let mut finished = false;
        let x = ncg(
            |x| A.t().dot(&(A.dot(&x) - &y)),
            400.0,
            x0.view_mut(),
            &Identity::new(),
            NITER + 6, // adding iterations shouldn't matter since
            10,        // we escape early
            |x, iter| {
                assert!(iter <= NITER);
                // println!("[{}, {}],", x[0], x[1]);
                let xk_star = xk_ref[iter];
                assert_eq!(x[0], xk_star[0]);
                assert_eq!(x[1], xk_star[1]);
                if iter == NITER {
                    finished = true
                }
                false
            },
        );
        assert_abs_diff_eq!(x, array![5., 5.], epsilon = 0.000001);
        assert_abs_diff_eq!(x0, array![5., 5.], epsilon = 0.000001);
        assert!(finished);
    }

    #[test]
    fn ncg_rand_quadratic() {
        let (M, N) = (100, 120);
        let A = Array::random((M, N), Normal::new(0., 10.).unwrap());
        let xtrue = Array::random((N,), Normal::new(0., 10.).unwrap());
        let y = A.dot(&xtrue);

        let mut x0 = Array::random((N,), Normal::new(0., 10.).unwrap());
        let L: f64 = A.svd(false, false).unwrap().1[0];
        let L: f64 = L.powf(2.);

        let x = ncg(
            |x| A.t().dot(&(A.dot(&x) - &y)),
            L,
            x0.view_mut(),
            &Identity::new(),
            300,
            50,
            |x, iter| {
                let cost: f64 = (A.dot(&x) - &y).norm_l2();
                let cost: f64 = cost.powf(2f64);
                println!("[{:?}] Cost = {:?}", iter, cost);
                false
            },
        );

        assert_abs_diff_eq!(A.dot(&x), y, epsilon = 0.000001);
    }

    #[test]
    fn ncg_els_simple_regression() {
        const NITER: usize = 3;
        let A = array![[10., 0.], [0., 20.]];
        let y = array![50., 100.];
        let mut x0 = array![19.0f32, 44.0f32];
        // Iterates check for regression
        let xk_ref: [[f32; 2]; NITER + 1] = [
            [19.0, 44.0],           // 0
            [15.478901, 4.7648964], // 1
            [5., 5.0000033],        // 2
            [5., 5.0000005],        // 3
        ];
        let mut finished = false;
        let x = ncg_els(
            &[A],
            &[(|x| &x - &y)],
            &[1.0],
            x0.view_mut(),
            &Identity::new(),
            NITER,
            10,
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
        assert_abs_diff_eq!(x, array![5., 5.], epsilon = 0.000001);
        assert_abs_diff_eq!(x0, array![5., 5.], epsilon = 0.000001);
        assert!(finished);
    }
}
