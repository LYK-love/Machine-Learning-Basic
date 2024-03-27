# Machine-Learning-Basic
The repo is the (re-)implementation of machine learning models and libraries.

# minigrad
Source: [minigrad](https://github.com/kennysong/minigrad) from Andrej Karpathy.

Requirements:
1. Use python=3.9
2. Install `graphviz` on your system. For example: `brew install graphviz` on MacOS.
3. `pip install graphviz`

There is also a `micrograd_from_scratch.ipynb` file which illustrate the usage of the libraries implemented in this repo.

# makemore
Source: [makemore](https://github.com/karpathy/makemore) from Andrej Karpathy, and the provided [notebook](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/makemore).

This dir contains my implementations for basic language models, where "basic" means I use plain neural networks, such as linear layers, MLPs, batch norms, etc., rather than other fancy models such as transformer.
In the last part, I built a [WaveNet](https://arxiv.org/pdf/1609.03499.pdf) which is a simple CNN language model. See my corresponding [blog post](https://lyk-love.cn/2024/03/23/wavenet-a-simple-illustration-of-neural-networks/)
There are notebook files in the directory to illustrate my understanding.


# GPT
Source: [nanogpt in Andrej Karpathy's lecture](https://github.com/karpathy/ng-video-lecture), and the provided [notebook](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing).

Currently I have implemented a nanoGPT, which is a simplied version GPT. The tokenizer will be implemented soon.

# VAE
Sources:
1. [VAE-tutorial](https://github.com/Jackson-Kang/Pytorch-VAE-tutorial)

This dir contains the implementations of Variational AutoEncoder(VAE) models.
Papers:
1. Variational AutoEncoder (VAE, D.P. Kingma et. al., 2013)
2. Vector Quantized Variational AutoEncoder (VQ-VAE, A. Oord et. al., 2017)
