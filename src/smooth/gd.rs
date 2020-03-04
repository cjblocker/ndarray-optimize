#![allow(non_snake_case)]

use super::smooth_line_search;
use crate::linop::LinearOperator;
use ndarray::prelude::*;
use ndarray::NdFloat; // includes LinalgScalar and ScalarOperand
use ndarray_linalg::Scalar;

/// Gradient Descent for L-Lipschitz Smooth Minimization
///
/// Also known as the unconstrained gradient method, or steepest descent.
///
/// Algorithm
/// ---------
/// ```math
/// x_{i+1} = x_i - \frac{h}{L} \nabla f(x_i)
/// ```
///
/// Parameters
/// ----------
/// - __grad:__      function that computes gradient g(x) of a convex cost function  
/// - __L:__         Lipschitz constant of cost function gradient  
/// - __h:__         normalized step size, typically in (0, 2) for convergence guarantees  
/// - __x0:__        initial guess  
/// - __maxiter:__   number of outer PSD iterations  
/// - __ninner:__    number of inner iterations of GD for line search  
/// - __callback:__  User-defined function to be evaluated with two arguments (x,iter).
///                   It is evaluated at (x0,0) and then after each iteration.
///                   If it returns True, the function terminates early.
pub fn gradient_descent<T: NdFloat>(
    grad: impl (Fn(ArrayView<T, Ix1>) -> Array<T, Ix1>),
    L: T,
    h: T,
    x0: ArrayViewMut1<'_, T>,
    maxiter: usize,
    mut callback: impl FnMut(ArrayView<T, Ix1>, usize) -> bool,
) -> ArrayViewMut1<'_, T> {
    let step = h / L;
    let mut x = x0;

    if callback(x.view(), 0) {
        return x;
    };
    for iter in 1..=maxiter {
        x.scaled_add(-step, &grad(x.view()));
        if callback(x.view(), iter) {
            break;
        };
    }
    x
}

/// Preconditioned Steepest Descent for L-Lipschitz Smooth Minimization
///
/// Algorithm
/// ---------
/// ```math
/// \begin{aligned}
/// d_i &= P \nabla f(x_i) \\
/// \alpha_i &= \mathrm{arg}\!\min_\alpha f(x_i - \alpha d_i ) \\
/// x_{i+1} &= x_i - \alpha_i d_i
/// \end{aligned}
/// ```
///
/// Parameters
/// ----------
/// - __grad:__      function that computes gradient g(x) of a convex cost function  
/// - __L:__         Lipschitz constant of cost function gradient  
/// - __x0:__        initial guess  
/// - __P:__         preconditioner  
/// - __maxiter:__   number of outer PSD iterations  
/// - __ninner:__    number of inner iterations of GD for line search  
/// - __callback:__  User-defined function to be evaluated with two arguments (x,iter).
///                   It is evaluated at (x0,0) and then after each iteration.
///                   If it returns True, the function terminates early.
pub fn steepest_descent<'a, T, S>(
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
    let mut x = x0;

    if callback(x.view(), 0) {
        return x;
    };
    for iter in 1..=maxiter {
        let dir = -P.apply_into(grad(x.view()));
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
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linop::Identity;
    use crate::smooth::nop;
    use approx::assert_abs_diff_eq;
    use test;

    #[test]
    fn test_gd_like_1d() {
        let a = 10.;
        let y = array![50.];
        let mut x0 = array![19.];
        let x = gradient_descent(|x| a * (a * &x - &y), 100.0, 1., x0.view_mut(), 20, nop);
        assert_abs_diff_eq!(x, array![5.]);
        assert_abs_diff_eq!(x0, array![5.]);
    }

    #[test]
    fn gradient_descent_simple_regression() {
        const NITER: usize = 100;
        let A = array![[10., 0.], [0., 20.]];
        let y = array![50., 100.];
        let mut x0 = array![19., 44.];
        // Reference from equivalnt NumPy code to
        // check for regressions.
        let xk_ref: [[f32; 2]; NITER / 20 + 1] = [
            [19.0, 44.0],             // 0
            [5.044396877288818, 5.0], // 20
            [5.000141143798828, 5.0], // 40
            [5.000000953674316, 5.0], // 60
            [5.000000953674316, 5.0], // 80
            [5.000000953674316, 5.0], // 100
        ];
        let mut finished = false;
        let x = gradient_descent(
            |x| A.t().dot(&(A.dot(&x) - &y)),
            400.0,
            1f32,
            x0.view_mut(),
            NITER,
            |x, iter| {
                assert!(iter <= NITER);
                if iter % 20 == 0 {
                    let xk_star = xk_ref[iter / 20];
                    assert_eq!(x[0], xk_star[0]);
                    assert_eq!(x[1], xk_star[1]);
                }
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

    #[test]
    fn steepest_descent_simple_regression() {
        const NITER: usize = 30;
        let A = array![[10., 0.], [0., 20.]];
        let y = array![50., 100.];
        let mut x0 = array![19., 44.];
        // Reference from equivalnt NumPy code to
        // check for regressions.
        let xk_ref: [[f32; 2]; NITER / 3 + 1] = [
            [19.0, 44.0],                           // 0
            [5.569559574127197, 4.956472396850586], // 3
            [5.023579120635986, 5.015727519989014], // 6
            [5.003462314605713, 4.999654293060303], // 9
            [5.000237941741943, 5.000135898590088], // 12
            [5.000040531158447, 4.999995231628418], // 15
            [5.000002384185791, 5.000001430511475], // 18
            [5.0, 5.0],                             // 21
            [5.0, 5.0],                             // 24
            [5.0, 5.0],                             // 27
            [5.0, 5.0],                             // 30
        ];
        let mut finished = false;
        let x = steepest_descent(
            |x| A.t().dot(&(A.dot(&x) - &y)),
            400.0,
            x0.view_mut(),
            &Identity::new(),
            NITER,
            10,
            |x, iter| {
                assert!(iter <= NITER);
                if iter % 3 == 0 {
                    let xk_star = xk_ref[iter / 3];
                    assert_eq!(x[0], xk_star[0]);
                    assert_eq!(x[1], xk_star[1]);
                }
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
