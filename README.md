\documentclass[11pt,a4paper]{article}

\usepackage[serbian]{babel}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{setspace}

\geometry{margin=2.5cm}
\setstretch{1.1}

\title{Ridge regresija}
\author{Filip Mar\v ci\'c}
\date{}

\begin{document}
\maketitle

\section*{Uvod}

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

\noindent
\textbf{Optimalni hiperparametar:}
\[
\lambda^* = \underline{45\hspace{2px}}
\]

\begin{figure}[h!]
\centering
\includegraphics[width=0.90\linewidth]{lambda.png}
\caption{Average CV RMSE u zavisnosti od $\lambda$ (senčeno: $\pm 1$ std preko ponavljanja).}
\label{fig:cv}
\end{figure}

\subsection*{Uticaj seed-a i randomizacije}

Seed za generisanje slučajnih permutacija inicijalno se bira nasumično, što znači da
se pri svakom pokretanju programa menjaju podele podataka na fold-ove.
Ova randomizacija direktno utiče na izračunate vrednosti RMSE i može dovesti do
različitog izbora optimalnog hiperparametra $\lambda^*$.

Zbog toga je \textbf{seed izuzetno važan}, jer značajno utiče na stabilnost i
reproduktivnost rezultata. Za konzistentne i uporedive rezultate preporučuje se
fiksiranje seed-a (npr.\ $seed = 42$).

\end{document}
