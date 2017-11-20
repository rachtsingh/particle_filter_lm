This is a rough set of ideas that I have during this project [remember to delete history when open-sourcing]

Let's say for now that we're staying fully Gaussian.

## Generative model
There's two main approaches that we could take here:

1. A template-like approach, where we generate a label sequence z_t from some prior, and then generate y from that sequence
2. A stepwise approach, where we generate a z_t, then feed the probability distribution over ys to generation of the next z

# Inference network
We need to find q(z | y), and we can do this in either the forward factorization, or the backwards factorization. For now, we might as well give the full biRNN information to the generation of z. Since this is Gaussian for now we sample.

# Particle filters
Let's wait on this until next week or so, but essentially the idea is that we'll maintain a set of S samples for each element of the batch
