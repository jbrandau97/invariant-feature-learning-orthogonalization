# Invariant Feature Learning via Orthogonalization

## Context

These days neural networks are able to learn and perform complex prediction tasks
with human-like accuracy. However, one of the challenges of using neural networks is
their tendency to extract spurious or unwanted relationships in the data set, leading
to poor generalization performance on new and unseen data. In the causal machine
learning literature this is sometimes called the ’Cows-on-the-Beach’ problem, as classifiers trained on pictures of cows in mainly alpine pastures fail to recognize cows
on beaches (see e.g. [Kaddour et al. (2022), Chapter 3](https://arxiv.org/abs/2206.15475)). Similarly, in the biomedical
field this is sometimes called ’confounding bias’, where classifiers trained on images
of diseases that often occur in the elderly will learn to detect the age of a patient as
an important disease predictor (see e.g. [Rao et al. (2017)](https://pubmed.ncbi.nlm.nih.gov/28143776/)). One possible solution is to
*deconfound* a model during training by forcing its learned features to be invariant
wrt. certain context variables.

The aim of this thesis is to test a new approach for invariant feature learning using
context variables based on the semi-structured networks (SSNs) proposed in [Rügamer,
Kolb, and Klein (2023)](https://www.tandfonline.com/doi/full/10.1080/00031305.2022.2164054). Consider a SSN model for responses $y$, structured covariates
$U$ (e.g. an image) and context variables $X$ (e.g. an age vector):

$$ \hat{y} = g(M_X f(U;\hat{\theta}) \hat{\gamma} + X \hat{\beta}). $$

Here $g(·)$ is a response function, $f(· ; \theta)$ are the output features of a (deep) neural
network with parameters $\theta$ and $(\beta, \gamma)$ are the parameters in the linear predictor.
Additionally, for $n$ obersvations the $n × n$ projection matrix $M_X = I_n − X(X^TX)^{−1}X^T$
projects the output features $f(· ; \theta)$ into a space orthogonal to the column space of
the context variables $X$, which allows the parameters $\beta$ to be uniquely identified.
While identification of $\beta$ is an important contribution of SNNs, the focus of this thesis
is on the orthogonalization via $M_X$, which should (in theory) force the network to
learn disentangled and invariant features $f(· ;\hat{\theta})$ with respect to the context variables
included in $X$. For the sake of this thesis it might therefore be enough to consider
the simplified model $\hat{y} = g(M_X f(U; \hat{\theta}) \hat{\gamma})$
that only includes the feature outputs and
orthogonalization.

## Goals of the Thesis

-  The primary goal of the thesis is to test a method for invariant feature learning
based on the orthogonalization proposed for SSNs in [Rügamer, Kolb, and
Klein (2023)](https://www.tandfonline.com/doi/full/10.1080/00031305.2022.2164054).
-  For this, the method should be implemented the deep learning framework
Pytorch and tested on a suitable simulated data set, possibly the ColoredMNIST data set used in [Arjovsky et al. (2019)](https://arxiv.org/abs/1907.02893). The results should be visualized in a way that demonstrates the invariance of the learned features, possibly using the methods described in [Simonyan, Vedaldi, and Zisserman (2013)](https://arxiv.org/abs/1312.6034).
-  A literature review should discuss similar methods that deal with invariant feature learning using context variables. They should be compared to the proposed method theoretically and in benchmark studies using e.g. simulated data sets.
