//! First Order Methods for L-Lipschitz Smooth Minimization
//!
//! For minimizing a "smooth" objective function, $`f`$, with an
//! $`L`$-Lipschitz continuos gradient, i.e.
//! ```math
//! \| \nabla f(x) - \nabla f(z) \|_2 \leq L \| x - z \|_2
//! ```
//! For twice-differentiable objective functions, $`L`$
//! represent an upper bound on the Hessian, $`\|\nabla^2 f(x)\|_2 \leq L`$,
//! but twice-differentiablity is not necessary.
//!
//! For example, for the least squares objective $`f(x) = \frac12 \|Ax-y\|_2^2`$, we have
//! $`\nabla^2 f(x) = A^HA \preceq \sigma_1(A)^2 I`$, and so $`L = \sigma_1(A)^2`$, the
//! largest singular value of $`A`$ squared.
//! This could also be shown from the definition.
//!
//! For more info, see [Lipschitz Continuity on Wikipedia](https://en.wikipedia.org/wiki/Lipschitz_continuity)

mod gd;
pub use gd::*;
mod ogm;
pub use ogm::*;
mod pcg;
pub use pcg::*;
mod bfgs;
pub use bfgs::*;

use ndarray::ArrayView;
use ndarray::NdFloat;

/// Do nothing function for optional user callback (returns false)
#[allow(clippy::needless_pass_by_value)]
pub fn nop<T, D>(_x: ArrayView<T, D>, _itr: usize) -> bool {
    false
}

/// 1D Gradient Descent
///
/// this method is faster than its ndarray counterpart
/// for the common case of doing 1D linesearches.
///
/// Parameters
/// ----------
/// __grad:__      function that computes gradient g(x) of a convex cost function  
/// __L:__         Lipschitz constant of cost function gradient  
/// __h:__         normalized step size, typically in (0, 2) for convergence guarantees  
/// __x0:__        initial guess  
/// __maxiter:__   number of outer PSD iterations  
pub fn smooth_line_search<T: NdFloat>(
    grad: impl (Fn(T) -> T),
    #[allow(non_snake_case)] L: T,
    h: T,
    x0: T,
    maxiter: usize,
) -> T {
    let step = h / L;
    let mut x = x0;

    for _iter in 1..=maxiter {
        x = x - step * grad(x);
    }
    x
}
