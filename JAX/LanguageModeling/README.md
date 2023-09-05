# Language Modeling


Language modeling (LM) is a natural language processing (NLP) task that determines the probability of a given sequence of words occurring in a sentence.

In an era where computers, smartphones and other electronic devices increasingly need to interact with humans, language modeling has become an indispensable technique for teaching devices how to communicate in natural languages in human-like ways.

But how does language modeling work? And what can you build with it? What are the different approaches, what are its potential benefits and limitations, and how might you use it in your business?

In this guide, you’ll find answers to all of those questions and more. Whether you’re an experienced machine learning engineer considering implementation, a developer wanting to learn more, or a product manager looking to explore what’s possible with natural language processing and language modeling, this guide is for you.

Here’s a look at what we’ll cover:

- Language modeling – the basics
- How does language modeling work?
- Use cases and applications
- Getting started


## Language modeling – the basics

### What is language modeling?

"*Language modeling is the task of assigning a probability to sentences in a language. […]
Besides assigning a probability to each sequence of words, the language models also assign a
probability for the likelihood of a given word (or a sequence of words) to follow a sequence
of words.*" Source: Page 105, [Neural Network Methods in Natural Language Processing](http://amzn.to/2wt1nzv), 2017.


### Types of language models

There are primarily two types of Language Models:

- Statistical Language Models: These models use traditional statistical techniques like N-grams, Hidden Markov Models (HMM), and certain linguistic rules to learn the probability distribution of words.
- Neural Language Models: They use different kinds of Neural Networks to model language, and have surpassed the statistical language models in their effectiveness. 

"*We provide ample empirical evidence to suggest that connectionist language models are
superior to standard n-gram techniques, except their high computational (training)
complexity.*" Source: [Recurrent neural network based language model](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf), 2010.

Given the superior performance of neural language models, we include in the container two popular state-of-the-art neural language models: BERT and Transformer-XL. 

### Why is language modeling important?

Language modeling is fundamental in modern NLP applications. It enables machines to understand qualitative information, and enables people to communicate with machines in the natural languages that humans use to communicate with each other. 

Language modeling is used directly in a variety of industries, including tech, finance, healthcare, transportation, legal, military, government, and more -- actually, you probably have just interacted with a language model today, whether it be through Google search, engaging with a voice assistant, or using text autocomplete features.


## How does language modeling work?

The roots of modern language modeling can be traced back to 1948, when Claude Shannon
published a paper titled "A Mathematical Theory of Communication", laying the foundation for information theory and language modeling. In the paper, Shannon detailed the use of a stochastic model called the Markov chain to create a statistical model for the sequences of letters in English text. The Markov models, along with n-gram, are still among the most popular statistical language models today. 

However, simple statistical language models have serious drawbacks in scalability and fluency because of its sparse representation of language. Overcoming the problem by representing language units (eg. words, characters) as a non-linear, distributed combination of weights in continuous space, neural language models can learn to approximate words without being misled by rare or unknown values.

Therefore, as mentioned above, we introduce two popular state-of-the-art neural language models, BERT and Transformer-XL, in Tensorflow and PyTorch. More details can be found in the [NVIDIA Deep Learning Examples Github Repository ](https://github.com/NVIDIA/DeepLearningExamples)


## Use cases and applications

### Speech Recognition

Imagine speaking a phrase to the phone, expecting it to convert the speech to text. How does
it know if you said "recognize speech" or "wreck a nice beach"? Language models help figure it out
based on the context, enabling machines to process and make sense of speech audio.


### Spelling Correction

Language-models-enabled spellcheckers can point to spelling errors and possibly suggest alternatives.


### Machine translation

Imagine you are translating the Chinese sentence "我在开车" into English. Your translation system gives you several choices:

- I at open car
- me at open car
- I at drive
- me at drive
- I am driving
- me am driving

A language model tells you which translation sounds the most natural.

## Getting started
NVIDIA provides examples for JAX models on [Rosetta](https://github.com/NVIDIA/JAX-Toolbox/tree/main/rosetta/rosetta/projects). These examples provide you with easy to consume and highly optimized scripts for both training and inferencing. The quick start guide at our GitHub repository will help you in setting up the environment using NGC Docker Images, download pre-trained models from NGC and adapt the model training and inference for your application/use-case.

These models are tested and maintained by NVIDIA, leveraging mixed precision using tensor cores on our latest GPUs for faster training times while maintaining accuracy.
