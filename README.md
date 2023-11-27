# Explorations of Contrast-Consistent Search

This repository is based entirely off the paper "Discovering Latent Knowledge in
Language Models Without Supervision" by Burns, Ye, Klein, and Steinhardt.

In particular, I'm using this as a testbed for other clustering techniques to
see whether they perform as well as Burns et al.'s simple MLP. 

In other words, I'm trying to figure out how much of the effectiveness of the
approach in the paper can be traced to its use of contrastive pairs alone and
how much relies specifically on using a neural net to perform classification.

The bulk of the work is in `main.py`.
