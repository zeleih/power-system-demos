# Analytical identification method

For one PMU sample increment, the retained-port network satisfies

\[
\Delta I_k=\bar{Y}\Delta U_k,
\qquad
\bar{Y}=G+jB,
\qquad
\bar{Y}=\bar{Y}^{\mathsf T}.
\]

For multiple samples, the objective is

\[
J=\sum_k\left\|\Delta I_k-\bar{Y}\Delta U_k\right\|_2^2.
\]

Because the admittance matrix is symmetric, each of \(G\) and \(B\) has
\(n(n+1)/2\) independent entries. Taking the derivative of the accumulated
objective with respect to every independent entry and setting the derivatives
to zero yields a square real system with \(n(n+1)\) unknowns.

The implementation does not retain a growing vectorized design matrix. It
accumulates

\[
C_U=\sum_k\Delta U_k^{\mathrm H}\Delta U_k,
\qquad
C_{UI}=\sum_k\Delta U_k^{\mathrm H}\Delta I_k,
\qquad
E_I=\sum_k\|\Delta I_k\|_2^2.
\]

After all available PMU batches have contributed, the independent complex
entries are represented by \(\theta=g+jb\), and the analytical equations are
solved in the paper's real block form,

\[
\begin{bmatrix}
\Re(H)&-\Im(H)\\
\Im(H)& \Re(H)
\end{bmatrix}
\begin{bmatrix}g\\b\end{bmatrix}
=
\begin{bmatrix}\Re(r)\\\Im(r)\end{bmatrix}.
\]

The estimated symmetric matrix is reconstructed from the independent entries.
The minimized residual is evaluated from the same accumulated terms, so the
historical PMU samples do not need to be revisited.

For reproducibility, the code also reports the rank and condition number of the
accumulated voltage-increment covariance. These are validation diagnostics for
the analytical equation system and do not introduce an iterative solver.

Implementation: [`src/gscr_demo/identification.py`](../src/gscr_demo/identification.py)
