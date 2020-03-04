//! Abstract Linear Operators and Adjoints
//! building upon ndarray_linalg::operator

use std::marker::PhantomData;

use ndarray::prelude::*;
use ndarray::{Data, DataMut, DataOwned, NdFloat};
pub use ndarray_linalg::diagonal::Diagonal;
pub use ndarray_linalg::operator::LinearOperator;
use ndarray_linalg::Scalar;
use num_traits::Float;

pub trait Adjoint<'a> {
    type Output;
    fn adj(&'a self) -> Self::Output;
}

impl<'a, A, S> Adjoint<'a> for ArrayBase<S, Ix2>
where
    A: 'a + Float,
    S: Data<Elem = A>,
{
    type Output = ArrayView<'a, A, Ix2>;

    fn adj(&'a self) -> Self::Output {
        self.t()
    }
}

impl<'a, A, Sa> Adjoint<'a> for Diagonal<Sa>
where
    A: 'a + Float,
    Sa: 'a + Data<Elem = A>,
{
    type Output = &'a Diagonal<Sa>;

    fn adj(&'a self) -> Self::Output {
        self
    }
}

#[derive(Default)]
pub struct Identity<A> {
    phantom: PhantomData<*const A>,
}

impl<A> Identity<A> {
    #[must_use]
    pub fn new() -> Identity<A> {
        Identity {
            phantom: PhantomData,
        }
    }
}

impl<A> LinearOperator for Identity<A>
where
    A: NdFloat + Scalar,
{
    type Elem = A;

    /// Apply operator out-place
    #[inline]
    fn apply<S>(&self, a: &ArrayBase<S, Ix1>) -> Array1<S::Elem>
    where
        S: Data<Elem = Self::Elem>,
    {
        a.to_owned()
    }

    /// Apply operator in-place
    #[inline]
    fn apply_mut<S>(&self, _a: &mut ArrayBase<S, Ix1>)
    where
        S: DataMut<Elem = Self::Elem>,
    {
    }

    /// Apply operator with move
    #[inline]
    fn apply_into<S>(&self, a: ArrayBase<S, Ix1>) -> ArrayBase<S, Ix1>
    where
        S: DataOwned<Elem = Self::Elem> + DataMut,
    {
        a
    }

    /// Apply operator to matrix out-place
    #[inline]
    fn apply2<S>(&self, a: &ArrayBase<S, Ix2>) -> Array2<S::Elem>
    where
        S: Data<Elem = Self::Elem>,
    {
        a.to_owned()
    }

    /// Apply operator to matrix in-place
    #[inline]
    fn apply2_mut<S>(&self, _a: &mut ArrayBase<S, Ix2>)
    where
        S: DataMut<Elem = Self::Elem>,
    {
    }

    /// Apply operator to matrix with move
    #[inline]
    fn apply2_into<S>(&self, a: ArrayBase<S, Ix2>) -> ArrayBase<S, Ix2>
    where
        S: DataOwned<Elem = Self::Elem> + DataMut,
    {
        a
    }
}

impl<'a, A: 'a> Adjoint<'a> for Identity<A> {
    type Output = &'a Identity<A>;

    fn adj(&'a self) -> Self::Output {
        self
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_linalg::diagonal::AsDiagonal;
    #[test]
    fn ndarray_adj() {
        let A = array![[1., 2.], [3., 4.]];
        let B = array![[1., 3.], [2., 4.]];
        assert!(A.adj() == B);
        assert!(A.adj().adj() == A);
    }

    #[test]
    fn diagonal_adj() {
        let d = array![1., 2., 3., 4.];
        let D = d.as_diagonal();
        assert!(D.adj().apply(&array![1., 1., 1., 1.]) == d);
    }

    #[test]
    fn identity() {
        let I = Identity::new();
        let d = array![1., 2., 3., 4.];

        assert!(I.apply(&d) == d);
        assert!(I.adj().apply(&d) == d);
    }
}
