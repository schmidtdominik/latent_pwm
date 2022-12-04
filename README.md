# latent_pwm

Learning latent policies and world models purely from observation (work in progress)

This project aims to learn policies and world models over a latent action space purely from observational data (no actions, no rewards). This can then be used in cases where an online environment is not available (so online RL can't be used), or offline data does not include rewards (so offline RL can't be used), or offline data does not include actions (so imitation learning, inverse RL can't be used). Additionally, it can be used for pretraining an exploratory policy from observations (for example youtube videos)

- `vqvae` contains a modified version of [this](https://github.com/MishaLaskin/vqvae) VQ-VAE implementation with some bugs fixed and other changes. The encoder is modified to also return the quantization indices. By only storing indices we can keep encoded datasets with 10B+ frames in memory. We use this VQ-VAE in a hierarchical setting as described in the paper.
- `cleanrl_datagen` contains modified versions of cleanrl's PPO implementation. This is used to train sample expert policies and roll these out in the environment to collect data.
- `alg_impala` and `alg_ppo` contain implementations of PPO and IMPALA to be used with multitask-dojo (these are currently broken due to changes in multitask-dojo).
- In the near future, code for training the VAE and world model based on ILPO will also be added.
