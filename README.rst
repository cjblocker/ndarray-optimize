ndarray-optimize
================
|build_status|_ |coverage|_

The ``ndarray-optimize`` crate provides a variety of optimization algorithms 
that can be used to minimize a function of an ``ndarray``.

It includes commonly used methods, such as:

- (L-)BFGS
- Nelder-Mead
- Conjugate Gradient
- FISTA

A goal of this crate is to provide functionality on par with popular
optimization packages in other languages, such as ``scipy.optimize`` in
Python, or ``Optim.jl`` for Julia.

This crate is in the early development stage and is actively changing.
The provided methods have been tested, but have not been tuned for
maximum performance or minimum memory usage. As practical benchmarks
are developed, more effort can be spent tuning the methods. If your
field has a canonical or common optimization problem that would
make a good benchmark, consider making a pull request!

Minimal ``rustc`` supported version is currently 1.39.0, 
but this repository targets rust stable

License
=======

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0 or the MIT license
http://opensource.org/licenses/MIT, at your
option. This file may not be copied, modified, or distributed
except according to those terms.


.. |build_status| image:: https://travis-ci.org/cjblocker/ndarray-optimize.svg?branch=master
.. _build_status: https://travis-ci.org/cjblocker/ndarray-optimize

.. |coverage| image:: https://coveralls.io/repos/github/cjblocker/ndarray-optimize/badge.svg?branch=master
.. _coverage: https://coveralls.io/github/cjblocker/ndarray-optimize?branch=master