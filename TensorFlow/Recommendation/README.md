# Recommender Systems


Recommender systems are a type of information filtering system that seeks to predict the
"rating" or "preference" a user would give to an item. (Source:
[Wikipedia](https://en.wikipedia.org/wiki/Recommender_system))

In an era where users have to navigate through an exponentially growing number of goods and services, recommender systems have become key in driving user engagement, teaching the internet services how to personalize experiences for users. They are ubiquitous and indispensable in commercial online platforms.

In this guide, you’ll find answers to how recommender systems work, how you might use it in your business, and more. Whether you’re an experienced machine learning engineer considering implementation, a developer wanting to learn more, or a product manager looking to explore what’s possible with recommender systems, this guide is for you.

Here is a look at what we will cover:

- Challenges and opportunities in recommender systems
- How does DL-based recommender systems work?
- Use cases and applications

## Challenges and opportunities in recommender systems

With the rapid growth in scale of industry datasets, deep learning (DL) recommender models have started to gain advantages over traditional methods by capitalizing on large amounts of training data. However, there are multiple challenges when it comes to performance of large-scale recommender systems solutions: 

- Huge datasets: Commercial recommenders are trained on huge datasets, often several terabytes in scale.
- Complex data preprocessing and feature engineering pipelines: Datasets need to be preprocessed and transformed into a form relevant to be used with DL models and frameworks. In addition, feature engineering creates an extensive set of new features from existing ones, requiring multiple iterations to arrive at an optimal solution.
- Input bottleneck: Data loading, if not well optimized, can be the slowest part of the training process, leading to under-utilization of high-throughput computing devices such as GPUs.
- Extensive repeated experimentation: The whole data engineering, training, and evaluation process is generally repeated many times, requiring significant time and computational resources.

To meet the computational demands for large-scale DL recommender systems training and inference, recommender-on-GPU solutions aim to provide fast feature engineering and high training throughput (to enable both fast experimentation and production retraining), as well as low latency, high-throughput inference.

Current DL–based models for recommender systems include the [Wide and
Deep](https://arxiv.org/abs/1606.07792) model, Deep Learning Recommendation Model
([DLRM](https://github.com/facebookresearch/dlrm)), neural collaborative filtering
([NCF](https://arxiv.org/abs/1708.05031)), Variational Autoencoder
([VAE](https://arxiv.org/abs/1802.05814)) for Collaborative Filtering, and
[BERT4Rec](https://arxiv.org/pdf/1904.06690.pdf), among others.

## How does DL-based recommender systems work?

In [NVIDIA Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples), we introduce several popular state-of-the-art DL-based recommender models in Tensorflow and PyTorch. 

As an example, we would like to start with discussing our reference implementation of DLRM. With DLRM, we systematically tackle the challenges mentioned by designing a complete DLRM pipeline, from data preparation to training to production inference. We provide ready-to-go Docker images for training and inference, data downloading and preprocessing tools, and Jupyter demo notebooks to get you started quickly. Also, trained models can be prepared for production inference in one simple step with our exporter tool. 

For more details on the model architectures, example code, and how to set to end-to-end data processing, training, and inference pipeline on GPU, please refer to the [DLRM developer blog](https://developer.nvidia.com/blog/optimizing-dlrm-on-nvidia-gpus/) and [NVIDIA GPU-accelerated DL model portfolio ](https://github.com/NVIDIA/DeepLearningExamples) under /PyTorch/Recommendation/DLRM.

In addition, DLRM forms part of NVIDIA [Merlin](https://developer.nvidia.com/nvidia-merlin), a framework for building high-performance, DL–based recommender systems. 

## Use cases and applications

### E-Commerce & Retail: Personalized Merchandising

Imagine a user has already purchased a scarf. Why not offer buying a hat that matches this hat, so that the look will be complete? This feature is often implemented by means of AI-based algorithms as “Complete the look”  or “You might also like” sections in e-commerce platforms like Amazon, Walmart, Target, and many others. 

On average, an intelligent recommender systems delivers a [22.66% lift in conversions rates](https://brandcdn.exacttarget.com/sites/exacttarget/files/deliverables/etmc-predictiveintelligencebenchmarkreport.pdf) for web products.

### Media & Entertainment: Personalized Content

AI based recommender engines can analyze the individual purchase behavior and detect patterns that will help provide a certain user with the content suggestions that will match his or her interests most likely. This is what Google and Facebook actively apply when recommending ads, or what Netflix does behind the scenes when recommending movies and TV shows.

### Personalized Banking

A mass market product that is consumed digitally by millions, banking is prime for recommendations. Knowing a customer’s detailed financial situation and their past preferences, coupled by data of thousands of similar users, is quite powerful.





