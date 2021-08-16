# Graph Computing Engine

- [PyTorch Geometric](#PyTorch-Geometric-PYG-Documentation-GitHub)
- [Deep Graph Library](#Deep-Graph-Library-DGL-Home-Page-Documentation-GitHub)
- [Graph Nets](#Graph-Nets-GitHub)
- [OpenNE](#OpenNE-GitHub)
- [PyTorch-BigGraph](#PyTorch-BigGraph-GitHub)
- [euler](#euler-GitHub)
- [StellarGraph](#StellarGraph-Home-Page-GitHub)
- [Spektral](#Spektral-Documentation-GitHub)
- [Jraph](#Jraph-Documentation-GitHub)
- [GeometricFlux.jl](#GeometricFluxjl-Home-Page-GitHub)

## PyTorch Geometric (PYG) [[Documentation]](https://pytorch-geometric.readthedocs.io/en/latest/) [[GitHub]](https://github.com/rusty1s/pytorch_geometric)
PyTorch Geometric is a geometric deep learning extension library for PyTorch.

It consists of various methods for deep learning on graphs and other irregular structures, also known as geometric deep learning, from a variety of published papers. In addition, it consists of an easy-to-use mini-batch loader for many small and single giant graphs, a large number of common benchmark datasets (based on simple interfaces to create your own), and helpful transforms, both for learning on arbitrary graphs as well as on 3D meshes or point clouds.

## Deep Graph Library (DGL) [[Home Page]](https://www.dgl.ai/) [[Documentation]](https://docs.dgl.ai/) [[GitHub]](https://github.com/dmlc/dgl)
Deep Graph Library (DGL) is a Python package built for easy implementation of graph neural network model family, on top of existing DL frameworks (currently supporting PyTorch, MXNet and TensorFlow). It offers a versatile control of message passing, speed optimization via auto-batching and highly tuned sparse matrix kernels, and multi-GPU/CPU training to scale to graphs of hundreds of millions of nodes and edges.


## Graph Nets [[GitHub]](https://github.com/deepmind/graph_nets)
Graph Nets is DeepMind's library for building graph networks in Tensorflow and Sonnet.

A graph network takes a graph as input and returns a graph as output. The input graph has edge- (E ), node- (V ), and global-level (u) attributes. The output graph has the same structure, but updated attributes. Graph networks are part of the broader family of "graph neural networks" (Scarselli et al., 2009).

## OpenNE [[GitHub]](https://github.com/thunlp/OpenNE)
OpenNE: An open source toolkit for Network Embedding. This repository provides a standard NE/NRL(Network Representation Learning）training and testing framework. In this framework, the OpenNE unifies the input and output interfaces of different NE models and provides scalable options for each model. Moreover, it implements typical NE models under this framework based on tensorflow, which enables these models to be trained with GPUs.

This toolkit is developed according to the settings of DeepWalk. The implemented or modified models include DeepWalk, LINE, node2vec, GraRep, TADW, GCN, HOPE, GF, SDNE and LE. the OpenNE will implement more representative NE models continuously according to our released NRL paper list. 

## PyTorch-BigGraph [[GitHub]](https://github.com/facebookresearch/PyTorch-BigGraph)
PyTorch-BigGraph (PBG) is a distributed system for learning graph embeddings for large graphs, particularly big web interaction graphs with up to billions of entities and trillions of edges.

PBG trains on an input graph by ingesting its list of edges, each identified by its source and target entities and, possibly, a relation type. It outputs a feature vector (embedding) for each entity, trying to place adjacent entities close to each other in the vector space, while pushing unconnected entities apart. Therefore, entities that have a similar distribution of neighbors will end up being nearby.

It is possible to configure each relation type to calculate this "proximity score" in a different way, with the parameters (if any) learned during training. This allows the same underlying entity embeddings to be shared among multiple relation types.

The generality and extensibility of its model allows PBG to train a number of models from the knowledge graph embedding literature, including TransE, RESCAL, DistMult and ComplEx.

PBG is not optimized for small graphs. If your graph has fewer than 100,000 nodes, consider using KBC with the ComplEx model and N3 regularizer. KBC produces state-of-the-art embeddings for graphs that can fit on a single GPU. Compared to KBC, PyTorch-BigGraph enables learning on very large graphs whose embeddings wouldn't fit in a single GPU or a single machine, but may not produce high-quality embeddings for small graphs without careful tuning.

## euler [[GitHub]](https://github.com/alibaba/euler)
Euler is a graph deep learning framework developed by Alibaba. Euler supports distributed training on large scale graph, which is common seen under practical usage scenarios. And it can represent complex  heterogeneous graph.

## StellarGraph [[Home Page]](https://www.stellargraph.io/) [[GitHub]](https://github.com/stellargraph/stellargraph)
The StellarGraph library offers state-of-the-art algorithms for graph machine learning, making it easy to discover patterns and answer questions about graph-structured data. It can solve many machine learning tasks:

* Representation learning for nodes and edges, to be used for visualisation and various downstream machine learning tasks;
* Classification and attribute inference of nodes or edges;
* Classification of whole graphs;
* Link prediction;
* Interpretation of node classification.

Graph-structured data represent entities as nodes (or vertices) and relationships between them as edges (or links), and can include data associated with either as attributes. For example, a graph can contain people as nodes and friendships between them as links, with data like a person’s age and the date a friendship was established. StellarGraph supports analysis of many kinds of graphs:

* homogeneous (with nodes and links of one type),
* heterogeneous (with more than one type of nodes and/or links)
* knowledge graphs (extreme heterogeneous graphs with thousands of types of edges)
* graphs with or without data associated with nodes
* graphs with edge weights

## Spektral [[Documentation]](https://graphneural.network/) [[GitHub]](https://github.com/danielegrattarola/spektral)
Spektral is a Python library for graph deep learning, based on the Keras API and TensorFlow 2. The main goal of this project is to provide a simple but flexible framework for creating graph neural networks (GNNs).

You can use Spektral for classifying the users of a social network, predicting molecular properties, generating new graphs with GANs, clustering nodes, predicting links, and any other task where data is described by graphs.

Spektral implements some of the most popular layers for graph deep learning, including:
* Graph Convolutional Networks (GCN)
* Chebyshev convolutions
* GraphSAGE
* ARMA convolutions
* Edge-Conditioned Convolutions (ECC)
* Graph attention networks (GAT)
* Approximated Personalized Propagation of Neural Predictions (APPNP)
* Graph Isomorphism Networks (GIN)
* Diffusional Convolutions

and many others (see convolutional layers).

## Jraph [[Documentation]](https://jraph.readthedocs.io/en/latest/) [[GitHub]](https://github.com/deepmind/jraph)
Jraph (pronounced "giraffe") is a lightweight library for working with graph neural networks in jax. It provides a data structure for graphs, a set of utilites for working with graphs, and a 'zoo' of forkable graph neural network models.

## GeometricFlux.jl [[Home Page]](https://fluxml.ai/GeometricFlux.jl/dev/) [[GitHub]](https://github.com/FluxML/GeometricFlux.jl)
GeometricFlux is a geometric deep learning library for Flux. This library aims to be compatible with packages from JuliaGraphs ecosystem and have support of CUDA GPU acceleration with CUDA. Message passing scheme is implemented as a flexbile framework and fused with Graph Network block scheme. GeometricFlux is compatible with other packages that are composable with Flux.
