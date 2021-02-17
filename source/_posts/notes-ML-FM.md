---
title: Applications of Machine Learning for Fluid Mechanics 
date: 2021-02-14 20:50:17
categories: Notes
tags: 
	- Machine Learning	
	- Deep Learning
	- Fluid Mechanics
---

This blog is the notes for a video called "[Machine Learning for Fluid Mechanics](https://www.youtube.com/watch?v=8e3OT2K99Kw&t=84s)", which is a brief introduction for a paper (Brunton, Steven L., Bernd R. Noack, and Petros Koumoutsakos. "Machine learning for fluid mechanics." Annual Review of Fluid Mechanics 52 (2020): 477-508.). If you want to know the details, please find the original video and paper.

<!-- more -->

# What is Machine Learning (ML)?

ML:

- Models from Data via Optimization

> Any sufficiently advanced technology is indistinguishable from magic.
>
> -- Arthur C. Clarke

Fluid dynamics tasks:

- Reduction
- Modeling
- Control
- Sensing
- Closure

Optimization problems:

- High-dimensional
- Non-linear
- Non-convex
- Multiscale

# What kind of ML is needed in science and engineering?

We need Interpretable and Generalizable Machine Learning in science and engineering field.

> Everything should be made as simple as possible, but not simpler.
>
> -- Albert Einstein

How to build a model like $F=ma$?

Features for ML in science and engineering:

- Sparse
- Low-dimensional
- Robust

# Schematic: ML + CFD

![Schematic](image-20210214173356481.png)



# Why ML could work?

Because patterns exist in fluid flow.

![pattern_FM](image-20210214173840928.png)

# Applications

## Fluid flow decomposition

PCA (Shallow, linear) -> Autoencoder (Deep)

![PCA_FM](image-20210214174003005.png)

![](image-20210214175022879.png)

## Denoise for Fluid Flow

![denoise](image-20210214174108194.png)

## Turbulence modeling

Paper: 

- Schlatter, Philipp, et al. "The structure of a turbulent boundary layer studied by numerical simulation." arXiv preprint arXiv:1010.4000 (2010).

- Duraisamy, Karthik, Gianluca Iaccarino, and Heng Xiao. "Turbulence modeling in the age of data." *Annual Review of Fluid Mechanics* 51 (2019): 357-377.

![](image-20210214174310749.png)

![](image-20210214174422276.png)

## ML_CFD solver

Paper:

- Ling, Julia, Andrew Kurzawski, and Jeremy Templeton. "Reynolds averaged turbulence modelling using deep neural networks with embedded invariance." Journal of Fluid Mechanics 807 (2016): 155-166.

Add physical constraints and achieve accurate and pyhsical.

## Super-resolution

Paper: 

- Erichson, N. Benjamin, Michael Muehlebach, and Michael W. Mahoney. "Physics-informed autoencoders for Lyapunov-stable fluid flow prediction." arXiv preprint arXiv:1905.10866 (2019).

Interpolation and Extrapolation

![](image-20210214174809740.png)

![](image-20210214174913278.png)

## Solve PDEs

![](image-20210214175304898.png)

# Beyond understanding: control

![](image-20210214182334199.png)

# Inspiration from biology

![](image-20210214182520461.png)
