# LEARNABLE POOLER

The aim of this project is to develop a new pooling operator that is parameterized such that backpropagation can be performed on it and such that it can have different behaviors based on how the parameters are optimized.

I studied the pooling operator presented in the review "A Comparison of PoolingMethods for Convolutional Neural Networks" published in the "Applied Science" on 29th August 2022.

The first part of the project is focused on implemententing Learnig Weighted Average Pooling, Gated Pooling Lp Pooling and S3 Pooling.
The second part of the project is focused on developing a variation on S3 Pooling that uses probability given by the Boltzmann's probability function, driven by the temperature parameter (learnable).
The third part of the project is focused on evaluating all the pooling operators described until now.

All the pooling operators are defined in the poolers.py file.
All the models are defined in the models.py file.
The body of the project is develop in the project.ipynb file.

In order to evaluate the different pooling operators MNIST is used.

This project is develop for academic purpose, since it is the final project for the Introduction to ML course (2024-2025).
