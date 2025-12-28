Na osnovu broja indexa \textbf{3318} dodeljeni zadatak je implementacija
\textbf{Ridge regresije} (linearne regresiju sa L2 regularizacijom), no u ovom domacem je napravljen promenljiva kojom proizvoljno mozemo menjati stepen (sa podrazumevanom vrednoscu 3).
\textbf{ponovljene k-fold
unakrsne validacije} na osnovu metrike RMSE.

\section*{Model i izbor hiperparametra}

Korišćen je model polinomne regresije sa Ridge regularizacijom, gde se ulazne
promenljive proširuju do stepena \textbf{3}. Ridge regularizacija se uvodi u normalne
jednačine u obliku:
\[
\mathbf{W} = (X^{T}X + \lambda R)^{-1} X^{T}y,
\]
pri čemu je $R$ identitetska matrica, uz izuzetak da se konstantni član (intercept)
ne regularizuje.

Hiperparametar koji se optimizuje je \textbf{$\lambda$}, koji kontroliše jačinu
regularizacije. Razmatrane su sledeće vrednosti:
\[
\lambda \in \{0, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 10\}.
\]

Izbor optimalnog hiperparametra izvršen je pomoću \textbf{repeated k-fold
cross-validation} sa parametrima $k=5$ i $10$ ponavljanja. Kao kriterijum kvaliteta
modela korišćena je srednja kvadratna greška korena (RMSE):
\[
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}.
\]

Optimalna vrednost hiperparametra određena je kao:
\[
\lambda^* = \arg\min_{\lambda} \; \text{Average CV RMSE}.
\]
