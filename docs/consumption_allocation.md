# Consumption Tilt Across Periods with Different Equivalence Scales

*Divorce branch — modeling note*

## Setup

Two periods with equivalence scales $s_1 = \sqrt{\text{hh\_size}_1}$, $s_2 = \sqrt{\text{hh\_size}_2}$ and per-equivalent felicity $f(C_t) = \big((C_t/s_t)^{1-\mu} - 1\big)/(1-\mu)$. The question is what multiplier, if any, sits in front of $f$ in the flow utility. Maximizing $u(C_1) + \beta u(C_2)$ subject to $C_1 + C_2/R = W$, the three candidates give:

**(a) No multiplier**, $u(C_t) = f(C_t)$:

$$\frac{C_2}{C_1} = (\beta R)^{1/\mu} \left(\frac{s_1}{s_2}\right)^{(1-\mu)/\mu}.$$

**(b) $\text{hh\_size}_t = s_t^2$ multiplier**, $u(C_t) = s_t^2 \, f(C_t)$:

$$\frac{C_2}{C_1} = (\beta R)^{1/\mu} \left(\frac{s_1}{s_2}\right)^{-(\mu+1)/\mu}.$$

**(c) $s_t = \sqrt{\text{hh\_size}_t}$ multiplier**, $u(C_t) = s_t \, f(C_t)$ — **the implemented choice**:

$$\frac{C_2}{C_1} = (\beta R)^{1/\mu} \left(\frac{s_1}{s_2}\right)^{-1}.$$

## Result

The sign of the exponent on $(s_1/s_2)$ determines which period the tilt favors.

- Under **(a)** the sign is $(1-\mu)/\mu$, which **depends on $\mu$**: for $\mu<1$ it is positive, so consumption tilts toward the *smaller*-household period (toward being alone). This is the pathology.
- Under **(b)** and **(c)** the exponent is negative for every $\mu>0$, so the tilt **never flips** — it always favors the larger household.

| $\mu$ | (a) no multiplier | (b) $s_t^2$ multiplier | (c) $s_t$ multiplier *(implemented)* | example ($s_1>s_2$) |
|---|---|---|---|---|
| $\mu > 1$ | larger scale | larger scale | larger scale | period 1 (partnered) |
| $\mu = 1$ | no tilt, $C_2/C_1=\beta R$ | larger scale | larger scale | period 1 (partnered) |
| $\mu < 1$ | **smaller scale** | larger scale | larger scale | period 2 (alone) |

*Direction of the equivalence-scale tilt in the Euler equation: which period's total consumption gets the extra share, beyond what $\beta R$ alone prescribes.*

The estimated model has $\mu_{\text{low}} = \mu_{\text{high}} = 0.8 < 1$ (`est_params_alg1_sparse.pkl`), so specification (a) would tilt consumption toward being alone — the wrong sign. Multiplying the felicity by $s_t = \sqrt{\text{hh\_size}_t}$ (specification (c), the implemented one) makes **per-equivalent consumption $C_t/s_t$ follow the standard Euler equation** $C_2/s_2 = (\beta R)^{1/\mu}\, C_1/s_1$, so total consumption always tilts toward the larger household regardless of $\mu$. This is the clean normalization: it removes the $\mu$-dependence of the tilt direction without the extra scale-squared force that (b) carries. See `marginal_utility_function_alive` in `utility_functions_add.py` / `utility_functions_cobb.py` for the corresponding marginal utility (the $s_t$ front factor cancels the $1/s_t$ from the chain rule, leaving $\text{marg} = (w\,C/s_t)^{-\mu}$ with the wealth-bookkeeping factor $w$).
