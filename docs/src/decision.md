
# The Incentive-Deterrence Model (Cont.)

## Basics
The dynamics of the system is,
```math
\left\{\begin{array}{ll}
    y_{t+1}    & = \varphi_{t+1}(x_t, y_t, \tau),         \\
    x_{t+1} & = \Gamma x_k - \hat\Gamma y_{t+1} + \lambda_t.
\end{array}\right.
```

In this part, we discuss using a rule to set $\tau \Rightarrow \tau_t$

At each $t+1$, we solve the following problem,
```math
\begin{aligned}
   \tau_{t+1} = \arg\min_\tau ~ & c_\tau^T\tau                  \\
    \textrm{s.t.} ~ & \|\tau - X_t^{-1}y_t\|_\infty \le \Delta
\end{aligned}
```
for multiple subpopulations,
```math
\begin{aligned}
\tau_{t+1} = \arg\min_\tau ~ & \sum_\tau (c_\tau^\nu)^T\tau^\nu     \\            
\textrm{s.t.} ~ & \|\tau^\nu - (X^\nu_t)^{-1}y^\nu_t\|_\infty \le \Delta
 \end{aligned}
```
## 

## Logistic Regression

Consider the logistic regression
```\math
f(x)=\frac{1}{m} \sum_{i=1}^m \log \left(1+e^{-b_i \cdot \cdot a_i^T x}\right)
```


## Fairness

> absolute fairness

The simplest fairness condition is,
```math
\tau^\nu_t \equiv \tau, \forall \nu
```