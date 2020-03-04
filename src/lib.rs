//! The `ndarray-optimize` crate provides a variety of optimization algorithms 
//! that can be used to minimize a function of an `ndarray`.
//! 
//! It includes commonly used methods, such as:
//! - (L)BFGS
//! - Nelder-Mead
//! - Conjugate Gradient
//! - FISTA
//! 
//! A goal of this crate is to provide functionality on par with popular
//! optimization packages in other languages, such as `scipy.optimize` in
//! Python, or `Optim.jl` for Julia.
//! 
//! This crate is in the early development stage and is actively changing.
//! The provided methods have been tested, but have not been tuned for
//! maximum performance or minimum memory usage. As practical benchmarks
//! are developed, more effort can be spent tuning the methods. If your
//! field has a canonical or common optimization problem that would
//! make a good benchmark, consider making a pull request!

#![cfg_attr(all(rustc_nightly, test), feature(test))]
#[cfg(all(rustc_nightly, test))]
extern crate test;

#[cfg(test)]
extern crate intel_mkl_src;

pub mod direct;
pub mod linop;
// pub mod nllsq;
pub mod prox;
pub mod smooth;
