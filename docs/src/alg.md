
# The Incentive-Deterrence Model

## Basics
The dynamics of the system is,
```math
\left\{\begin{array}{ll}
    y_{t+1}    & = \varphi_{t+1}(x_t, y_t, \tau),         \\
    x_{t+1} & = \Gamma x_k - \hat\Gamma y_{t+1} + \lambda_t.
\end{array}\right.
```
The main fixed-point iteration is at,
```@docs
Criminos.F
```

We define two data structs:
- the system struct, $\Psi$, to store constants
- the state variable, $z$, for $(x,y)$ in the math, and attributes to enable efficient computations 😍 
```@docs
Criminos.BidiagSys
```
```@docs
Criminos.MarkovState
```

## Dynamics of $x$
The $x$ part is simple,
```@docs
Criminos.Fₓ
```
```math
x_{t+1} = \Gamma x_t - \hat\Gamma y_t + \lambda_t.
```
In this view, we use $\Psi$
```math
\quad \Gamma =\mathrm{diag}(\gamma), \quad \hat \Gamma =
\begin{bmatrix}
    \gamma_0  & 0        & \cdots        & 0            \\
    -\gamma_0 & \gamma_1 & \cdots        & 0            \\
    \vdots    & \ddots   & \ddots        & \vdots       \\
    0         & \cdots   & -\gamma_{N-2} & \gamma_{N-1},
\end{bmatrix}.
```
such that we call the `bi-diagonal system`. 
The associated "potential" or Lynapunov function is,
```math
u(x, y) = \frac{1}{2}(x)^T(I - \Gamma)x - (\lambda)^T x + (x)^T \hat{\Gamma} y
```

## Adversarial Energy, and Best Response

Note $\varphi$ of $y$ follows the utility/potential/energy $\omega$.

```@docs
Criminos.w
```
```@docs
Criminos.mixed_in_gnep_best!
```
```@docs
Criminos.mixed_in_gnep_grad!
```

we take arguments 
```
∇₀, H₀, ∇ₜ, Hₜ, _... = args
```
for any $y,\tau$, compute $c$ from the exogenous arrivals
```math
c = Q\lambda
```

> the deterrence & incentives
```math
\begin{aligned}
\nabla^2\omega &= \operatorname{diag}(\tau) \cdot H_t \cdot \operatorname{diag}(\tau) + H_0 \\
\nabla\omega &= (\nabla^2\omega) y + \underbrace{\nabla_t \tau - \nabla_0 c}_{g} + \hat \Gamma^Tx
\end{aligned}
```
and finally
```math
\omega(y,\tau) = \frac{1}{2}y^T (\nabla^2\omega) y + y^T \left(\nabla_t \tau - \nabla_0 c\right) + y^T\hat \Gamma^Tx
```

!!! note
    One can see $\tau$ is crucial to compute the second-order derivatives. If $\tau = 0$, this reduce to $H_0, \nabla_0 c$...
    This can be taken as `intrinsic` property of a subpopulation

## Extension to Multiple Subpopulation
For multiple population, $\nu = 0,..., M-1$, now $\omega$ has multiple blocks, $y^0, \ldots, y^\nu, \ldots, y^{M-1}$, stick to the parameterization,
```math
g^\nu = \nabla^\nu_t \tau^\nu - \nabla^\nu_0 c^\nu
```
Then the energy can we rewritten as,
```math
\begin{aligned}
    \omega(y, x, \tau) := 
    y^T \begin{bmatrix}
            (\hat{\Gamma}^0)^T x^0 \\
            \vdots \\
            (\hat{\Gamma}^{M-1})^T x^{M-1} \\
        \end{bmatrix} +
    y^T \begin{bmatrix}
        g^0 \\
        \vdots \\
        g^{M-1} \\
    \end{bmatrix} +

    \frac{1}{2} y^T \begin{bmatrix}
                       H^1       & \cdots & H^{1,M-1} \\
                       \vdots    & \ddots & \vdots    \\
                       H^{M-1,1} & \cdots & H^{M-1}   \\
                     \end{bmatrix} y
\end{aligned}
```
!!! note 
    If $H_t^{i,j} \neq 0$, then the recidivism of $i$ is affected by the recidivism of $j$.

Population is still independent,
```math
\forall \nu, \quad u^\nu(x^\nu, y_k^\nu) = \frac{1}{2}(x^\nu)^T(I - \Gamma^\nu)x^\nu - (\lambda^\nu)^T x^\nu + (x^\nu)^T \hat{\Gamma}^\nu y_k^\nu
```



## Index

```@index
```
