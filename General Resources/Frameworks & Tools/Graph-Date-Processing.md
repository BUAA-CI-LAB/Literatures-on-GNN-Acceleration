# Graph Data Processing Tools

- [Metis](#Metis-Home-Page-Download)
- [GraphX](#GraphX-Home-Page-Download)
- [PowerGraph](#PowerGraph-GitHub)

## Metis [[Home Page]](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) [[Download]](http://glaros.dtc.umn.edu/gkhome/metis/metis/download)

---
METIS is a set of serial programs for **partitioning graphs, partitioning finite element meshes**, and **producing fill reducing orderings for sparse matrices**. The algorithms implemented in METIS are based on the multilevel recursive-bisection, multilevel k-way, and multi-constraint partitioning schemes developed in our lab.

METIS's key features are the following:

[1] **Provides high quality partitions!**

Experiments on a large number of graphs arising in various domains including finite element methods, linear programming, VLSI, and transportation show that METIS produces partitions that are consistently better than those produced by other widely used algorithms. The partitions produced by METIS are consistently 10% to 50% better than those produced by spectral partitioning algorithms.

[2] **It is extremely fast!**

Experiments on a wide range of graphs has shown that METIS is one to two orders of magnitude faster than other widely used partitioning algorithms. Graphs with several millions of vertices can be partitioned in 256 parts in a few seconds on current generation workstations and PCs.

[3] **Produces low fill orderings!**

The fill-reducing orderings produced by METIS are significantly better than those produced by other widely used algorithms including multiple minimum degree. For many classes of problems arising in scientific computations and linear programming, METIS is able to reduce the storage and computational requirements of sparse matrix factorization, by up to an order of magnitude. Moreover, unlike multiple minimum degree, the elimination trees produced by METIS are suitable for parallel direct factorization. Furthermore, METIS is able to compute these orderings very fast. Matrices with millions of rows can be reordered in just a few seconds on current generation workstations and PCs.

## GraphX [[Home Page]](http://spark.apache.org/graphx/) [[Download]](http://spark.apache.org/downloads.html)

---
GraphX is a new component in Spark for graphs and graph-parallel computation. At a high level, GraphX extends the Spark RDD by introducing a new Graph abstraction: a directed multigraph with properties attached to each vertex and edge. To support graph computation, GraphX exposes a set of fundamental operators (e.g., subgraph, joinVertices, and aggregateMessages) as well as an optimized variant of the Pregel API. In addition, GraphX includes a growing collection of graph algorithms and builders to simplify graph analytics tasks.

[1] **Flexibility**

GraphX unifies ETL, exploratory analysis, and iterative graph computation within a single system. You can view the same data as both graphs and collections, transform and join graphs with RDDs efficiently, and write custom iterative graph algorithms using the Pregel API.

[2] **Speed**

GraphX competes on performance with the fastest graph systems while retaining Spark's flexibility, fault tolerance, and ease of use.

[3] **Algorithms**

In addition to a highly flexible API, GraphX comes with a variety of graph algorithms, many of which were contributed by our users.

## PowerGraph [[GitHub]](https://github.com/jegonzal/PowerGraph)

GraphLab PowerGraph is a graph-based, high performance, distributed computation framework written in C++.

The GraphLab PowerGraph academic project was started in 2009 at Carnegie Mellon University to develop a new parallel computation abstraction tailored to machine learning. GraphLab PowerGraph 1.0 employed shared-memory design. In GraphLab PowerGraph 2.1, the framework was redesigned to target the distributed environment. It addressed the difficulties with real-world power-law graphs and achieved unparalleled performance at the time. In GraphLab PowerGraph 2.2, the Warp System was introduced and provided a new flexible, distributed architecture around fine-grained user-mode threading (fibers). The Warp System allows one to easily extend the abstraction, to improve optimization for example, while also improving usability.

GraphLab PowerGraph is the culmination of 4-years of research and development into graph computation, distributed computing, and machine learning. GraphLab PowerGraph scales to graphs with billions of vertices and edges easily, performing orders of magnitude faster than competing systems. GraphLab PowerGraph combines advances in machine learning algorithms, asynchronous distributed graph computation, prioritized scheduling, and graph placement with optimized low-level system design and efficient data-structures to achieve unmatched performance and scalability in challenging machine learning tasks.

Related is GraphChi, a spin-off project separate from the GraphLab PowerGraph project. GraphChi was designed to run very large graph computations on just a single machine, by using a novel algorithm for processing the graph from disk (SSD or hard drive) enabling a single desktop computer (actually a Mac Mini) to tackle problems that previously demanded an entire cluster. For more information, see https://github.com/GraphChi.
