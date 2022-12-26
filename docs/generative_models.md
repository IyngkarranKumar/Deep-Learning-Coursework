# Generative models

Aim to learn the true distribution of the data, $p_{true}(X)$. If a data point $x$ is a function of some latent variables $Z=(z_1,z_2...z_n)$, then $P(Z)$ will suffice.

https://dlvu.github.io/slides/dlvu.lecture06.pdf
https://openai.com/blog/glow/



### On the merits of different generative models

A discussion of the relative merits between generative models. Based on the following blog post https://openai.com/blog/glow/. We begin by introducing some key concepts in generative modelling.
<br>

#### Latent variable inference

- Flow-based models learn a bijective mapping between ambient and latent variables. Thus, each data point has a unique latent representation. Thus, we can perform exact inference.

#### log-likelihood

Some geneartive models are more *expressive* than others, meaning they can learn more complex (and thus potentially more accurate) models of the data. For example, the VAE is more expressive than flow models (non-linear transformation vs linear/invertible), meaning it achieves higher log-likelihood.

#### Inference and synthesis efficiency

This refers to the ease of inferring latent variables/sampling datapoint given a latent variable. Consider a RNN. It requires outputs of previous $(t-1)$ steps for $t$th output, which leads to slower synthesis.

#### Time-space complexity
Apparently reversible NNs (flow models) have memory costs that is constant in depth rather than linear. I.e - some of these models should be less expensive (thus faster?) to train.s

## Models

#### VAEs

#### GANs
- Have strong performance on image synthesis
- Don't explicity map to latent space (but why does this matter?)
- Unstable training

#### Autoregressive models

#### Flow-based models

#### Energy models

#### Implicit models (?)
