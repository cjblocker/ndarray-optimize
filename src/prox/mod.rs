//! Minimization for Composite Functions consisting of
//! L-smooth _f_ and non-smooth prox-friendly _g_
//!
//! This includes common objective functions such as the LASSO
//! as well as smooth constrained methods, as a projection is
//! the proximal operator of a constraint set.

mod pogm;
pub use pogm::*;

mod fista;
pub use fista::*;