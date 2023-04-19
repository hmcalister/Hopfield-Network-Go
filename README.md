# Hopfield Network Go
#### Author: Hayden McAlister

![Badge for the Go build+test workflow](https://github.com/hmcalister/Hopfield-Network-Go/actions/workflows/go.yml/badge.svg?branch=main)

## Introduction

This project is an investigation into implementing the Hopfield network (and some other supporting methods) in Go using gonum as a linear algebra backend. This project is intended to be clean and extensible, as well as blazing fast and scalable with CPU cores via threading. 

In future it may be interesting to try and port this project to use a different backend project - one that leverages linear algebra on CUDA to scale instead with the GPU.


## Why Go?

Go was chosen for this project for the following reasons:

- It was found to be fast (see the profiling and testing [in this repository](https://github.com/hmcalister/Linear-Algebra-Profiling) - be sure to checkout the dashboard!)

- Tensorflow was found to scale much better by leveraging the GPU, but ensuring the code continued to scale required awkward vectorized methods that were prone to bugs.

- Rust was found to scale nearly as well as Go on the CPU, and has nicer memory safety. However, multithreading proved to be difficult, and the implementation did not continue very far past the initial experiments. Check out the [Rust implementation](https://github.com/hmcalister/Hopfield-Network-Rust).

- Go was found to scale very slightly better on the CPU, and after the [initial implementation](https://github.com/hmcalister/Hopfield-Network-Go) the language was found to be a nicer fit. Higher velocity development wins the day!
