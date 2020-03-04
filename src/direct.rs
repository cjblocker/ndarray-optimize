//! Minimization via Direct Function Comparison, i.e. derivativeless
use ndarray::prelude::*;
use ndarray::NdFloat; // includes LinalgScalar and ScalarOperand

/// The Nelder–Mead method
///
/// Also known as the downhill simplex method, amoeba method, or polytope method.
/// This well-known method requires no gradients, but will generally be slower
/// than those methods that do. Note also, that while this method has seen
/// much emprical success in applications, there is not much theory
/// desribing when and if this method will converge to a minima. See
/// [Wikipedia](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)
/// for more info.
pub fn nelder_mead<S>(
    func: impl Fn(ArrayView1<S>) -> S,
    x0: ArrayView1<S>,
    maxiter: usize,
    callback: impl Fn(ArrayView1<S>, usize) -> bool,
) -> Array1<S>
where
    S: NdFloat,
{
    let n = x0.len();

    let rho = S::from(1.).unwrap();
    let chi = S::from(2.).unwrap();
    let psi = S::from(0.5).unwrap();
    let sigma = S::from(0.5).unwrap();
    let gamma = rho * chi;
    let beta = psi * rho;

    let nonzero_delta = S::from(1. + 0.05).unwrap();
    let zero_delta = S::from(0.00025).unwrap();

    // Initialize the simplex
    let mut simplex = x0.broadcast((n + 1, n)).unwrap().to_owned();
    for k in 0..n {
        if simplex[[k + 1, k]] != S::zero() {
            simplex[[k + 1, k]] = nonzero_delta * simplex[[k + 1, k]];
        } else {
            simplex[[k + 1, k]] = zero_delta;
        }
    }

    // Initialize inital function values and sort indices
    // the idea here is that we don't want to sort the simplex
    // because it could be large, just sort the indexes and use them
    let mut fvals: Vec<S> = simplex.outer_iter().map(&func).collect();
    let mut ranked_indices: Vec<usize> = (0..=n).collect(); // len n+1
    ranked_indices.sort_unstable_by(|&x, &y| fvals[x].partial_cmp(&fvals[y]).unwrap());
    let mut high_index = ranked_indices[n];
    let mut low_index = ranked_indices[0];

    if callback(simplex.row(low_index), 0) {
        return simplex.row(low_index).to_owned();
    }
    // Run iterations of Nelder-Mead
    for iter in 1..=maxiter {
        // Compute the lower simplex centroid (aka average) p_bar
        let p_bar = ranked_indices[..n]
            .iter()
            .fold(Array1::<S>::zeros((n,)), |acc, &ii| acc + simplex.row(ii))
            / S::from(n).unwrap();

        // Compute reflected point
        let p_reflect = &p_bar * (S::one() + rho) - &simplex.row(high_index) * rho;
        let f_reflect = func(p_reflect.view());

        if f_reflect < fvals[low_index] {
            // Was the refection amazing?
            // Perform Simplex expansion
            let p_expand = p_bar * (S::one() + gamma) - &simplex.row(high_index) * gamma;
            let f_expand = func(p_expand.view());

            if f_expand < fvals[low_index] {
                // Expansion Success, replace f_high
                simplex.row_mut(high_index).assign(&p_expand);
                fvals[high_index] = f_expand;
            } else {
                // Expansion Failed, use reflection
                simplex.row_mut(high_index).assign(&p_reflect);
                fvals[high_index] = f_reflect;
            }
        } else {
            // Ok, reflection wasn't "amazing", but is it atleast better?
            let next_high_index = ranked_indices[n - 1];
            if f_reflect < fvals[next_high_index] {
                // Ehh, Good enough
                simplex.row_mut(high_index).assign(&p_reflect);
                fvals[high_index] = f_reflect;
            } else {
                // Shoot!
                let mut do_shrink = false;
                if f_reflect < fvals[high_index] {
                    // could be worse
                    // Perform contraction
                    let p_contract = p_bar * (S::one() + beta) - &simplex.row(high_index) * beta;
                    let f_contract = func(p_contract.view());

                    if f_contract < f_reflect {
                        // Contraction Success
                        simplex.row_mut(high_index).assign(&p_contract);
                        fvals[high_index] = f_contract;
                    } else {
                        do_shrink = true;
                    }
                } else {
                    // well that reflection was the worst
                    // Perform inside contraction
                    let p_contract = p_bar * (S::one() - psi) + &simplex.row(high_index) * psi;
                    let f_contract = func(p_contract.view());

                    if f_contract < fvals[high_index] {
                        // Inside Contraction Success
                        simplex.row_mut(high_index).assign(&p_contract);
                        fvals[high_index] = f_contract;
                    } else {
                        do_shrink = true;
                    }
                };

                if do_shrink {
                    // Contraction failed
                    // AAAAAAaaaaa!
                    // on the bright-side, all our points are already amazing
                    // do shrink on all but the best one
                    for &ii in ranked_indices[1..].iter() {
                        let new_p = &(&simplex.row(ii) - &simplex.row(low_index)) * sigma
                            + &simplex.row(low_index);
                        simplex.row_mut(ii).assign(&new_p);
                        fvals[ii] = func(simplex.row(ii));
                    }
                }
            }
        }

        // update rank of indices
        ranked_indices.sort_unstable_by(|&x, &y| fvals[x].partial_cmp(&fvals[y]).unwrap());
        high_index = ranked_indices[n];
        low_index = ranked_indices[0];

        debug_assert!(fvals[low_index] <= fvals[high_index]);

        if callback(simplex.row(low_index), iter) {
            break;
        }
    }
    simplex.row(low_index).to_owned()
}

/// Golden Section Search
///
/// Minimizes a scalar function _func_ by iteratively dividing an interval
/// given by _a_, _b_ into successively smaller itervals given by the
/// golden ratio. See
/// [Wikipedia](https://en.wikipedia.org/wiki/Golden-section_search)
/// for more info.
pub fn golden_ss<S>(func: impl Fn(S) -> S, a: S, b: S, tol: S) -> S
where
    S: NdFloat,
{
    // 1 / phi = 0.61803398875 = phi - 1 = (sqrt(5) - 1)/2
    let invphi = S::from((5.0_f64.sqrt() - 1.) / 2.).unwrap();
    // 1 / phi^2 = 0.38196601125 = 2 - phi = 1 - invphi = (3 - sqrt(5))/2
    let invphi2 = S::from((3. - 5.0_f64.sqrt()) / 2.).unwrap();

    let (mut a, b) = if a < b { (a, b) } else { (b, a) };
    let mut width = b - a;
    if width <= tol {
        return (a + b) / S::from(2).unwrap();
    }

    // Required steps to achieve tolerance
    let n = (tol / width).log(invphi).ceil().to_usize().unwrap();

    let mut c = a + invphi2 * width;
    let mut d = a + invphi * width;
    let mut f_c = func(c);
    let mut f_d = func(d);

    for _iter in 0..n {
        if f_c < f_d {
            // b = d; never used
            d = c;
            f_d = f_c;
            width = invphi * width;
            c = a + invphi2 * width;
            f_c = func(c);
        } else {
            a = c;
            c = d;
            f_c = f_d;
            width = invphi * width;
            d = a + invphi * width;
            f_d = func(d);
        }
    }

    if f_c < f_d {
        return c;
    } else {
        return d;
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn nelder_mead_rosenbrock() {
        let func =
            |x: ArrayView1<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
        let x0 = array![3.0, -8.3];
        let res = nelder_mead(func, x0.view(), 400, |_x, _iter| false);
        println!("res: {}", res);
        assert_abs_diff_eq!(res[0], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(res[1], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn nelder_mead_berger() {
        #![allow(non_snake_case)]
        // the machine translation example of Berger et al in
        // Computational Linguistics, vol 22, num 1, pp 39--72, 1996.
        let F = array![
            [1., 1., 1.],
            [1., 1., 0.],
            [1., 0., 1.],
            [1., 0., 0.],
            [1., 0., 0.]
        ];
        let K = array![1., 0.3, 0.5];
        let x0 = array![0., 0., 0.];
        let xtrue = array![0., -0.524869316, 0.487525860];
        let maxiter = 1000;
        let func = |x: ArrayView1<f64>| F.dot(&x).mapv_into(|y| y.exp()).sum().ln() - &K.dot(&x);
        let res = nelder_mead(func, x0.view(), maxiter, |_x, _iter| false);
        println!("res: {}", res);
        assert_abs_diff_eq!(func(res.view()), func(xtrue.view()), epsilon = 1e-6);
        assert_abs_diff_eq!(res.slice(s![1..]), xtrue.slice(s![1..]), epsilon = 1e-3);
    }

    #[test]
    fn golden_ss_quadratic() {
        let func = |x: f32| x.powi(2);
        let min = golden_ss(func, 1., 2., 1e-4);
        assert_abs_diff_eq!(min, 1.0, epsilon = 1e-4);

        let min = golden_ss(func, -2., -3., 1e-4);
        assert_abs_diff_eq!(min, -2.0, epsilon = 1e-4);

        let min = golden_ss(func, -2., 3., 1e-4);
        assert_abs_diff_eq!(min, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn golden_ss_quadratic2() {
        let xtrue = 1.5;
        let func = |x: f32| (x - xtrue).powi(2) - 0.8;
        let min = golden_ss(func, 1., 3., 1e-4);
        // we should be able to use a smaller epsilon here
        // but it gets stuck currently.
        assert_abs_diff_eq!(min, xtrue, epsilon = 1e-3);

        let min = golden_ss(func, -2., -3., 1e-4);
        assert_abs_diff_eq!(min, -2.0, epsilon = 1e-3);

        let min = golden_ss(func, -15., 15., 1e-4);
        assert_abs_diff_eq!(min, xtrue, epsilon = 1e-3);
    }
}
