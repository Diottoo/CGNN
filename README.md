# Conformalized-GNN-with-Conditional-Guarantees
For the base(uncalibrated) GNN model, consider 'GAT', 'GCN', 'GraphSAGE', 'SGC'.

Let $G=(\mathcal{V}, \mathcal{E})$ be a graph, where $\mathcal{V}$ and $\mathcal{E}$ are the set of nodes of the graph and the corresponding set of edges. The data-structured dataset $(\textbf{X}, \mathcal{Y}) = \{(X_i, Y_i)\}_{i=1}^n$ contains feature vectors $\textbf{X}_v\in\mathbb{R}^d$ associated to each node $v\in \mathcal{V}$, and $Y_v\in\mathcal{Y}$ is the label of node $v$. Each $(X_i, Y_i)$ is assumed to be drawn i.i.d from the distribution $P$. We empirically apply our frame to two kinds of tasks, node classification and node regression. We currently only consider the transductive setting where each node learns the representation by utilizing all the neighborhood information. In the last section we will discuss the possible ways for the inductive setting.

The obtained prediction sets are expected to meet 3 competing goals: (1) distribution-free: does not rely on specific assumptions about the probability distribution or structure of the data, (2) be valid in finite samples, (3) conditional coverage: satisfy $\mathbb{P}(Y_{n+1}\in \hat{C}(X_{n+1})|X_{n+1}=x)=1-a$. 

It has been demonstrated that it is impossible to satisfy these conditions simultaneously (Vovk, 2012; Barber et al., 2020). A widely adopted approach that comes close to achieving this is conformal prediction, while it only gives marginal coverage rather than conditional ones $\mathbb{P}(Y_{n+1}\in \hat{C}(X_{n+1}))=1-a$.

**Proposition 1**
P(Y_{n+1} ∈ Ĉ(X_{n+1}) | X_{n+1} = x) = 1 - α, for all x  
⟺  
E[f(X_{n+1})(1{Y_{n+1} ∈ Ĉ(X_{n+1})} - (1 - α))] = 0, for all measurable f.

*Proof*

P(Y_{n+1} ∈ Ĉ(X_{n+1}) | X_{n+1} = x) = 1 - α  
⟺ E[1{Y_{n+1} ∈ Ĉ(X_{n+1})} | X_{n+1} = x] = 1 - α  
⟺ E[f(X_{n+1}) ⋅ E[1{Y_{n+1} ∈ Ĉ(X_{n+1})} | X_{n+1} = x]] = E[f(X_{n+1}) ⋅ (1 - α)]  
⟺ E[f(X_{n+1}) ⋅ (1{Y_{n+1} ∈ Ĉ(X_{n+1})} - (1 - α))] = 0


To make it achievable, we can relax the requirement by only ensuring coverage over "a selected class of functions F" rather than over "all measurable f." 

Consider the relaxed coverage objective, this subset could potentially be infinite, but it is less complex than the set of all measurable functions. For example, if \( \mathcal{F} \) is \( \{x \mapsto 1\} \), this yields \textbf{marginal coverage}—i.e., on average across all inputs, the coverage requirement is met. If we choose more complex classes \( \mathcal{F} \), the resulting coverage can bridge the gap between marginal coverage (broad, average-level coverage) and full conditional coverage (coverage for every possible value of \( X \)).

\textbf{Conformal Prediction.} As we discussed above, conformal prediction, which gives marginal coverage predictions, is a specific trivial of the general method. Here we focus on the split conformal prediction method. Given a predefined miscoverage rate \(\alpha \in [0, 1]\), CP proceeds in three steps:\\
i) \underline{Non-conformity Scores}: CP firstly computes a heuristic measure of uncertainty, which is the non-conformity score here \( V : \mathcal{X} \times \mathcal{Y} \to \mathbb{R} \). For example, in classification, \( V(x, y) \) could be the predicted probability of class \( y \), or in regression, \( V(x, y) = |y - \hat{\mu}(x)| \) for a predictor \( \hat{\mu} : \mathcal{X} \to \mathcal{Y} \). Given dataset \(\{(X_i, Y_i)\}_{i=1}^n\), this step will generate a new set of non-conformity scores $\mathcal{V}=\{V_1, \ldots, V_n\}$\\
ii) \underline{Quantile Computation}: 
CP then calculates the \( 1 - \alpha \) quantile of the non-conformity scores. \( \hat{\eta} = \text{quantile}(\mathcal{V}, (1 - \alpha)(1 + \frac{1}{n})) \)\\
iii) \underline{Prediction Set/Interval Construction}: For a new test point \( X_{n+1} \), CP constructs a prediction set or interval \( C(X_{n+1}) = \{ y \in \mathcal{Y} : V(X_{n+1}, y) \leq \hat{\eta} \} \). If the data points \(\{Z_i\}_{i=1}^{n+1} := \{(X_i, Y_i)\}_{i=1}^{n+1}\) are exchangeable, then \( V_{n+1} := V(X_{n+1}, Y_{n+1}) \) is exchangeable with \(\{V_i\}_{i=1}^n\), given the predictor \( \hat{\mu} \).


Thus, \(\hat{C}(X_{n+1})\) contains the true label with a predefined coverage rate \cite{ref43}: 
\[
\mathbb{P}(Y_{n+1} \in C(X_{n+1})) = \mathbb{P}\left( V_{n+1} \geq \text{Quantile}(\{V_1, \dots, V_{n+1}\}, 1 - \alpha) \right) \geq 1 - \alpha,
\]
due to the exchangeability of \(\{V_i\}_{i=1}^{n+1}\). This framework applies to any choice of non-conformity score, making it score-agnostic. CF-GNN is similarly non-conformity score-agnostic. However, for demonstration, we focus on two popular scores, described in detail below.

\begin{algorithm}
\caption{SplitConformal($D, V, \alpha$)}\label{alg:cap}
\begin{algorithmic}
\State Let \( \hat{\eta}\) be the smallest value such that:
\[
\sum_{i=1}^{n} \mathbf{1}[V(x_i, y_i) \leq  \hat{\eta}] \geq (1 - \alpha)(n + 1)
\]
\hfill

\State i.e. \( \hat{\eta}\) is an empirical \(\left\lceil (1 - \alpha) \frac{(n + 1)}{n} \right\rceil\) quantile of \(D\), \( \hat{\eta} = \text{quantile}(\mathcal{V}, (1 - \alpha)(1 + \frac{1}{n})) \)
\hfill

\State Output the function that can give prediction interval to new data point:
\[
C(x) = \left\{ \hat{y} : V(x, \hat{y}) \leq \hat{\eta} \right\}
\]
\end{algorithmic}
\end{algorithm}

\newpage
Recall the objective

\[
\mathbb{E}\left[f(X_{n+1}) \left( \mathbf{1}\{Y_{n+1} \in \hat{C}(X_{n+1})\} - (1 - \alpha) \right) \right] = 0, \quad \forall f \in \mathcal{F}.  \tag{1}
\]

$\mathcal{F}=\{\theta; \theta(x)=1\}$ is the case of split conformal prediction, $\mathcal{F}=\{\theta; \theta(x) \in \mathbb{R}\}$ is a more general case that gives marginal coverage for the prediction sets. A even more general function class can be formally expressed as $\mathcal{F} = \{\Phi(\cdot)^\top \beta : \beta \in \mathbb{R}^d\}$, which is a class of linear functions over the basis $\Phi : \mathcal{X} \to \mathbb{R}^d$. The goal is to construct a $\hat{C}$ satisfying (1) for this choice of $\mathcal{F}$.\\


A universal method that relax the restriction to the basis $\Phi(\cdot)$ is considering covariate shifts generated by softmax. \\

For classification case, the prediction set $\hat{C}_{smx}(\cdot)$ has a basis
\begin{align*}
    &\Phi(\cdot) = \text{span}\{\phi_1(\cdot), \ldots, \phi_d(\cdot)\}\\
    & \phi_k = \mathbb{P}(k|x) = \frac{e^{\hat{y}_k/T}}{\sum_{k'=1}^d e^{\hat{y}_{k'}/T}}, \;\;\;\; k \in \{1, 2, \ldots, d\}
\end{align*}
where T is the tempurature of softmax, and $\hat{y}_k$ are logits, which are raw predictions of the original model $\text{GNN}_o$. Note that $\log\mathbb{P}(k|x) \propto  \hat{y}_k $ \\

In the proof of proposition 1, the last step is obtained by the law of iterated expectations that relax conditional expectations to unconditional expectations. Utilizing the same law, the objective (1) can be reversely modified to a conditional probability of $Y$, \[\mathbb{P}(Y_{n+1} \in \hat{C}(X_{n+1})|Y_{n+1} =y) =1-\alpha \tag{2}\]

```bibtex
@article{huang2023conformalized_gnn,
  title={Uncertainty quantification over graph with conformalized graph neural networks},
  author={Huang, Kexin and Jin, Ying and Candes, Emmanuel and Leskovec, Jure},
  journal={NeurIPS},
  year={2023}
}
