# Hydrogen-line emissivity methods

For the analytic H$\alpha$ branch at $T_{\rm QK}\geq3000\,{\rm K}$ and for
the all-cell QUOKKA H I result, the electron fraction is inferred from the mean
molecular weight. For an ideal gas,

$$
\frac{1}{\mu}
=
\frac{(\gamma-1)m_{\rm H}e_{\rm int}}
{\rho k_{\rm B}T_{\rm QK}},
$$

and therefore

$$
x_e\equiv\frac{n_e}{n_{\rm H}}
=
\frac{\mu^{-1}-X-Y/4}{X},
$$

where $X$ and $Y$ are the hydrogen and helium mass fractions, respectively.
No upper or lower limit is imposed on the value of $x_e$ obtained from this
expression. The electron number density is

$$
n_e=x_e n_{\rm H}.
$$

Here $n_{\rm H}=n_{\rm HI}+n_{\rm H^+}$ is the total hydrogen-nuclei number
density.

## H$\alpha$ emissivity

The H$\alpha$ volume emissivity is calculated as

$$
\epsilon_{\rm H\alpha}
=
0.45\,h\nu_{\rm H\alpha}\,\alpha_{\rm B}(T_{\rm QK})
\,n_e n_{\rm H^+},
$$

where

$$
\alpha_{\rm B}(T)
=
2.54\times10^{-13}
T_4^{-0.8163-0.0208\ln T_4}
\;{\rm cm^3\,s^{-1}},
\qquad
T_4\equiv\frac{T}{10^4\,{\rm K}}.
$$

The proton density is determined from $x_e$. When $x_e\leq1$, the hydrogen
ionization fraction is assumed to equal the electron fraction:

$$
x_{\rm H^+}\equiv\frac{n_{\rm H^+}}{n_{\rm H}}=x_e,
\qquad
n_{\rm H^+}=x_e n_{\rm H}.
$$

When $x_e>1$, all hydrogen is assumed to be ionized and the additional
electrons are attributed to helium and metals. In this case,

$$
x_{\rm H^+}
=
\frac{n_{\rm H^+}}{n_{\rm H}}
=
\frac{n_{\rm H^+}}{n_{\rm HI}+n_{\rm H^+}}
=
\frac{n_{\rm H^+}}{n_{\rm H^+}}
=1,
$$

and hence $n_{\rm H^+}=n_{\rm H}$. The proton density can therefore be written
as the piecewise function

$$
n_{\rm H^+}
=
\begin{cases}
x_e n_{\rm H}, & x_e\leq1,\\
n_{\rm H}, & x_e>1.
\end{cases}
$$

## H I 21-cm emissivity

The optically thin H I 21-cm volume emissivity is calculated twice, using two
independent estimates of the neutral-hydrogen density:

$$
\epsilon_{\rm HI}
=
\frac{3}{4}\,n_{\rm HI}A_{10}h\nu_{21}.
$$

For the DESPOTIC result, every cell uses the neutral-hydrogen density returned
by the DESPOTIC chemistry table. For the QUOKKA result, every cell uses
$T_{\rm QK}$ in the mean-molecular-weight inversion. When $x_e\leq1$, the
neutral-hydrogen density is

$$
\begin{aligned}
n_{\rm HI}
&=n_{\rm H}-n_{\rm H^+}\\
&=\left(1-\frac{n_{\rm H^+}}{n_{\rm H}}\right)n_{\rm H}\\
&=(1-x_{\rm H^+})n_{\rm H}\\
&=(1-x_e)n_{\rm H}.
\end{aligned}
$$

When $x_e>1$, all hydrogen is assumed to be ionized, so $n_{\rm HI}=0$. The
QUOKKA neutral-hydrogen density is therefore

$$
n_{\rm HI}
=
\begin{cases}
(1-x_e)n_{\rm H}, & x_e\leq1,\\
0, & x_e>1.
\end{cases}
$$
