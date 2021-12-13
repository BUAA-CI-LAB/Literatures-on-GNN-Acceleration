# Literature on Graph Neural Networks Acceleration

A reading list for deep graph learning acceleration, including but not limited to related research on software and hardware level. The list covers related papers, conferences, tools, books, blogs, courses and other resources. We have a [team of Maintaners](#Maintainers) responsible for maintainance, meanwhile also welcome contributions from anyone.

Literatures in this page are arranged from a classification perspective, including the following topics:
- [Literature on Graph Neural Networks Acceleration](#literature-on-graph-neural-networks-acceleration)
  - [Hardware Acceleration for Graph Neural Networks](#hardware-acceleration-for-graph-neural-networks)
  - [System Designs for Deep Graph Learning](#system-designs-for-deep-graph-learning)
  - [Algorithmic Acceleration for Graph Neural Networks](#algorithmic-acceleration-for-graph-neural-networks)
  - [Surveys and Performance Analysis on Graph Learning](#surveys-and-performance-analysis-on-graph-learning)
  - [Maintainers](#maintainers)

Click [here](./By-Time.md) to view these literatures in a reverse chronological order. You can also find [Related Conferences](./General%20Resources/Conference.md), [Graph Learning Tools](./General%20Resources/Frameworks%20%26%20Tools/), [Learning Materials on GNNs](./General%20Resources/Learning%20Materials) and Other Resources in [General Resources](./General%20Resources).

---
## Hardware Acceleration for Graph Neural Networks
* [**EuroSys 2021**] Accelerating Graph Sampling for Graph Machine Learning Using GPUs. *Jangda, et al.* [[Paper]]( https://doi.org/10.1145/3447786.3456244)
* [**ASICON 2019**] An FPGA Implementation of GCN with Sparse Adjacency Matrix. *Ding et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/8983647)
* [**MICRO 2021**] AWB-GCN: A Graph Convolutional Network Accelerator with Runtime Workload Rebalancing. *Geng et al.* [[Paper]](https://ieeexplore.ieee.org/document/9252000)
* [**DAC 2021**] BlockGNN: Towards Efficient GNN Acceleration Using Block-Circulant Weight Matrices. *Zhou et al.* [[Paper]](https://arxiv.org/abs/2104.06214)
* [**FCCM 2021**] BoostGCN: A Framework for Optimizing GCN Inference on FPGA. *Zhang et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9444065)
* [**TCAD 2021**] Cambricon-G: A Polyvalent Energy-efficient Accelerator for Dynamic Graph Neural Networks. *Song et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9326339)
* [**ICCAD 2020**] DeepBurning-GL: an automated framework for generating graph neural network accelerators. *Liang et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9256539)
* [**TC 2020**] EnGN: A High-Throughput and Energy-Efficient Accelerator for Large Graph Neural Networks. *Liang et al.* [[Paper]](https://ieeexplore.ieee.org/document/9161360/)
* [**Access 2020**] FPGAN: An FPGA Accelerator for Graph Attention Networks With Software and Hardware Co-Optimization. *Yan et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9195849)
* [**HPCA 2021**] GCNAX: A Flexible and Energy-efficient Accelerator for Graph Convolutional Neural Networks. *Li et al.* [[Paper]](https://doi.org/10.1109/HPCA51647.2021.00070)
* [**SC 2020**] GE-SpMM: General-Purpose Sparse Matrix-Matrix Multiplication on GPUs for Graph Neural Networks. Huang, et al. [[Paper]](https://doi.org/10.1109/sc41405.2020.00076)
* [**DAC 2021**] GNNerator: A Hardware/Software Framework for Accelerating Graph Neural Networks. *Stevens et al.* [[Paper]](https://arxiv.org/abs/2103.10836)
* [**CCIS 2020**] GNN-PIM: A Processing-in-Memory Architecture for Graph Neural Networks. *Wang et al.* [[Paper]](https://www.semanticscholar.org/paper/GNN-PIM%3A-A-Processing-in-Memory-Architecture-for-Wang-Guan/1d03e4bebc9cf3c3fdd9204504d92b20d97d1fdf)
* [**ATC 2021**] GLIST: Towards In-Storage Graph Learning. *Li, et al.* [[Paper]](www.usenix.org/conference/atc21/presentation/li-cangyuan)
* [**FPGA 2020**] GraphACT: Accelerating GCN Training on CPU-FPGA Heterogeneous Platforms. *Zeng et al.* [[Paper]](https://arxiv.org/abs/2001.02498)
* [**arXiv 2020**] GRIP: A Graph Neural Network Accelerator Architecture. *Kiningham et al.* [[Paper]](https://arxiv.org/abs/2007.13828)
* [**CAL 2021**] Hardware Acceleration for GCNs via Bidirectional Fusion. *Li et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9425440)
* [**DAC 2020**] Hardware Acceleration of Graph Neural Networks. *Auten, et al.* [[Paper]](https://doi.org/10.1109/dac18072.2020.9218751)
* [**ASAP 2020**] Hardware Acceleration of Large Scale GCN Inference. *Zhang et al.*[[Paper]](https://ieeexplore.ieee.org/document/9153263)
* [**HPCA 2020**] HyGCN: A GCN Accelerator with Hybrid Architecture. *Yan, et al.* [[Paper]](https://doi.org/10.1109/HPCA47549.2020.00012)
* [**arXiv 2021**] LW-GCN: A Lightweight FPGA-based Graph Convolutional Network Accelerator. *Tao, et al.*  [[Paper]](https://arxiv.org/abs/2111.03184)
* [**MICRO 2021**] Point-X: A Spatial-Locality-Aware Architecture for Energy-Efficient Graph-Based Point-Cloud Deep Learning. *Zhang, et al.*  [[Paper]](https://dl.acm.org/doi/10.1145/3466752.3480081)
* [**DATE 2021**] ReGraphX: NoC-Enabled 3D Heterogeneous ReRAM Architecture for Training Graph Neural Networks. *Arka, et al.* [[Paper]](https://doi.org/10.23919/DATE51398.2021.9473949)
* [**TCAD 2021**] Rubik: A Hierarchical Architecture for Efficient Graph Neural Network Training. *Chen et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9428002)
* [**ICPADS 2020**] S-GAT: Accelerating Graph Attention Networks Inference on FPGA Platform with Shift Operation. *Yan et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9359183)
* [**EuroSys 2021**] Tesseract: distributed, general graph pattern mining on evolving graphs. *Bindschaedler, et al.* [[Paper]](https://doi.org/10.1145/3447786.3456253)
* [**ICA3PP 2020**] Towards a Deep-Pipelined Architecture for Accelerating Deep GCN on a Multi-FPGA Platform. *Cheng et al.* [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-60245-1_36)
* [**SCIS 2021**] Towards Efficient Allocation of Graph Convolutional Networks on Hybrid Computation-In-Memory Architecture. *Chen et al.* [[Paper]](https://link.springer.com/article/10.1007/s11432-020-3248-y)
* [**arXiv 2021**] VersaGNN: a Versatile accelerator for Graph neural networks. *Shi et al.* [[Paper]](https://arxiv.org/abs/2105.01280)
* [**arXiv 2021**] ZIPPER: Exploiting Tile- and Operator-level Parallelism for General and Scalable Graph Neural Network Acceleration. *Zhang et al.* [[Paper]](https://arxiv.org/abs/2107.08709v1)

---
## System Designs for Deep Graph Learning
* [**CLUSTER 2021**] 2PGraph: Accelerating GNN Training over Large Graphs on GPU Clusters.*Zhang et al.* [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9556026)
* [**JPDC  2021**] Accurate, efficient and scalable training of Graph Neural Networks *Zeng et al.* [[Paper]](https://www.sciencedirect.com/science/article/pii/S0743731520303579)
* [**KDD  2019**] AliGraph: a comprehensive graph neural network platform. *Yang et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3292500.3340404) [[GitHub]](https://github.com/alibaba/graph-learn)
* [**OSDI  2021**] Dorylus: Affordable, Scalable, and Accurate GNN Training with Distributed CPU Servers and Serverless Threads. *Thorpe et al.* [[Paper]](https://arxiv.org/abs/2105.11118) [[GitHub]](https://github.com/uclasystem/dorylus)
* [**EuroSys 2021**] DGCL: an efficient communication library for distributed GNN training. *Cai et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3447786.3456233)
* [**IA3  2020**] DistDGL: Distributed Graph Neural Network Training for Billion-Scale Graphs. *Zheng et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9407264)
* [**CoRR  2019**] Deep Graph Library: Towards Efficient and Scalable Deep Learning on Graphs. *Wang* [[Paper]](https://arxiv.org/abs/1909.01315v2) [[GitHub]](https://github.com/dmlc/dgl/) [[Home Page]](https://www.dgl.ai/)
* [**TPDS  2021**] Efficient Data Loader for Fast Sampling-Based GNN Training on Large Graphs. *Bai et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9376972)
* [**TPDS  2020**] EDGES: An Efficient Distributed Graph Embedding System on GPU Clusters. *Yang et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9272876)
* [**GNNSys 2021**] FedGraphNN: A Federated Learning System and Benchmark for Graph Neural Networks.*He et al.* [[Paper]](https://gnnsys.github.io/papers/GNNSys21_paper_3.pdf) [[Poster]](https://gnnsys.github.io/posters/GNNSys21_poster_3.pdf)
* [**EuroSys  2021**] FlexGraph: a flexible and efficient distributed framework for GNN training. *Wang et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3447786.3456229) [[GitHub]](https://github.com/snudatalab/FlexGraph)
* [**ICCAD 2020**] fuseGNN: accelerating graph convolutional neural network training on GPGPU. *Chen et al.* [[Paper]](https://ieeexplore.ieee.org/document/9256702) [[GitHub]](https://github.com/apuaaChen/gcnLib)
* [**SC 2020**] FeatGraph: A Flexible and Efficient Backend for Graph Neural Network Systems. *Hu et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9355318) [[Github]](https://github.com/amazon-research/FeatGraph)
* [**ICLR  2019**] Fast Graph Representation Learning with PyTorch Geometric. *Fey et al.* [[Paper]](https://arxiv.org/abs/1903.02428) [[GitHub]](https://github.com/rusty1s/pytorch_geometric) [[Documentation]](https://pytorch-geometric.readthedocs.io/en/latest/)
* [**IPDPS  2021**] FusedMM: A Unified SDDMM-SpMM Kernel for Graph Embedding and Graph Neural Networks. *Rahman et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9460486)
* [**GNNSys 2021**] Graphiler: A Compiler for Graph Neural Networks.*Xie et al.* [[Paper]](https://gnnsys.github.io/papers/GNNSys21_paper_10.pdf) [[Poster]](https://gnnsys.github.io/posters/GNNSys21_poster_10.pdf)
* [**OSDI   2021**] GNNAdvisor: An Adaptive and Efﬁcient Runtime System for GNN Acceleration on GPUs *Wang et al.* [[Paper]](https://www.usenix.org/system/files/osdi21-wang-yuke.pdf)
* [**AccML  2020**] GIN : High-Performance, Scalable Inference for Graph Neural Networks. *Fu et al.* [[Paper]](https://workshops.inf.ed.ac.uk/accml/papers/2020/AccML_2020_paper_6.pdf)
* [**JPDC  2021**] High performance GPU primitives for graph-tensor learning operations. *Zhang et al.* [[Paper]](https://www.sciencedirect.com/science/article/abs/pii/S0743731520304007)
* [**GNNSys 2021**] IGNNITION: A framework for fast prototyping of Graph Neural Networks.*Pujol-Perich et al.* [[Paper]](https://gnnsys.github.io/papers/GNNSys21_paper_4.pdf) [[Poster]](https://gnnsys.github.io/posters/GNNSys21_poster_4.pdf)
* [**MLSys  2020**] Improving the Accuracy, Scalability, and Performance of  Graph Neural Networks with Roc. *Jia* [[Paper]](https://www-cs.stanford.edu/people/matei/papers/2020/mlsys_roc.pdf)
* [**GNNSys 2021**] Load Balancing for Parallel GNN Training.*Su et al.* [[Paper]](https://gnnsys.github.io/papers/GNNSys21_paper_18.pdf) [[Poster]](https://gnnsys.github.io/posters/GNNSys21_poster_18.pdf)  
* [**ATC  2019**] NeuGraph: Parallel Deep Neural Network Computation on Large Graphs. *Ma et al.* [[Paper]](https://www.usenix.org/conference/atc19/presentation/ma)
* [**arXiv  2021**] PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models. *Rozemberczki et al.* [[Paper]](https://arxiv.org/abs/2104.07788) [[GitHub]](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)
* [**SoCC  2020**] PaGraph: Scaling GNN training on large graphs via computation-aware caching. *Lin et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3419111.3421281)
* [**IPDPS  2020**] Pcgcn: Partition-centric processing for accelerating graph convolutional network. *Tian et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9139807/)
* [**SysML 2019**] PyTorch-BigGraph: A Large-scale Graph Embedding System. *Lerer et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3292500.3340404) [[GitHub]](https://github.com/facebookresearch/PyTorch-BigGraph)
* [**arXiv 2018**] Relational inductive biases, deep learning, and graph networks. *Battaglia et al.* [[Paper]](https://arxiv.org/abs/1806.01261) [[GitHub]](https://github.com/deepmind/graph_nets)
* [**EuroSys 2021**] Seastar: vertex-centric programming for graph neural networks. *Wu, et al.* [[Paper]](https://doi.org/10.1145/3447786.3456247)


---
## Algorithmic Acceleration for Graph Neural Networks

* [**GNNSys 2021**] Adaptive Filters and Aggregator Fusion for Efficient Graph Convolutions.*Tailor et al.* [[Paper]](https://arxiv.org/abs/2104.01481) [[GitHub]](https://github.com/shyam196/egc)
* [**PMLR 2021**] A Unified Lottery Ticket Hypothesis for Graph Neural Networks. *Chen et al.* [[Paper]](http://proceedings.mlr.press/v139/chen21p.html)
* [**PVLDB 2021**] Accelerating Large Scale Real-Time GNN Inference using Channel Pruning. *Zhou et al.* [[Paper]](https://doi.org/10.14778/3461535.3461547)
* [**IPDPS 2019**] Accurate, efficient and scalable graph embedding. *Zeng et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/8820993)
* [**CVPR  2021**] Bi-GCN: Binary Graph Convolutional Network. *Wang et al.* [[Paper]](https://arxiv.org/abs/2010.07565)
* [**CVPR  2021**] Binary Graph Neural Networks. *Bahri et al.* [[Paper]](https://arxiv.org/abs/2012.15823)
* [**GLSVLSI 2021**] Co-Exploration of Graph Neural Network and Network-on-Chip Design Using AutoML. *Manu et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3453688.3461741)
* [**ICLR  2021**] Degree-Quant: Quantization-Aware Training for Graph Neural Networks. *Tailor et al.* [[Paper]](https://arxiv.org/abs/2008.05000)
* [**SC 2021**] Efficient scaling of dynamic graph neural networks.*Chakaravarthy et al.* [[Paper]](https://doi.org/10.1145/3458817.3480858)
* [**GNNSys 2021**] Efficient Data Loader for Fast Sampling-based GNN Training on Large Graphs.*Bai et al.* [[Paper]](https://gnnsys.github.io/papers/GNNSys21_paper_8.pdf) [[Poster]](https://gnnsys.github.io/posters/GNNSys21_poster_8.pdf)
* [**GNNSys 2021**] Effiicent Distribution for Deep Learning on Large Graphs.*Hoang et al.* [[Paper]](https://gnnsys.github.io/papers/GNNSys21_paper_16.pdf) [[Poster]](https://gnnsys.github.io/posters/GNNSys21_poster_16.pdf)
* [**ICLR 2021 Open Review**] FGNAS: FPGA-AWARE GRAPH NEURAL ARCHITECTURE SEARCH. *Qing et al.* [[Paper]](https://openreview.net/pdf?id=cq4FHzAz9eA)
* [**arXiv 2021**] GNNSampler: Bridging the Gap between Sampling Algorithms of GNN and Hardware. *Liu et al.* [[Paper]](https://arxiv.org/abs/2108.11571v1)
* [**ICCAD 2021**] G-CoS: GNN-Accelerator Co-Search Towards Both Better Accuracy and Efficiency. *Zhang et al.* [[Paper]](https://arxiv.org/pdf/2109.08983.pdf)
* [**arXiv 2020**] Learned Low Precision Graph Neural Networks. *Zhao et al.* [[Paper]](https://www.euromlsys.eu/pub/zhao21euromlsys.pdf)
* [**RTAS 2021**] Optimizing Memory Efficiency of Graph Neural Networks on Edge Computing Platforms. *Zhou et al.* [[Paper]](https://arxiv.org/abs/2104.03058) [[GitHub]](https://github.com/BUAA-CI-Lab/GNN-Feature-Decomposition)
* [**SC 2020**] Reducing Communication in Graph Neural Network Training. *Tripathy et al.* [[Paper]](https://arxiv.org/abs/2005.03300) [[GitHub]](https://github.com/PASSIONLab/gnn_training)
* [**ICTAI 2020**] SGQuant: Squeezing the Last Bit on Graph Neural Networks with Specialized Quantization. *Feng et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9288186)
* [**KDD  2020**] TinyGNN: Learning Efficient Graph Neural Networks. *Yan et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3394486.3403236)


---
## Surveys and Performance Analysis on Graph Learning
* [**GNNSys 2021**] Analyzing the Performance of Graph Neural Networks with Pipe Parallelism.*T. Dearing et al.* [[Paper]](https://gnnsys.github.io/papers/GNNSys21_paper_12.pdf) [[Poster]](https://gnnsys.github.io/posters/GNNSys21_poster_12.pdf)
* [**IJCAI 2021**] Automated Machine Learning on Graphs: A Survey. *Zhang et al.* [[Paper]](https://arxiv.org/abs/2103.00742)
* [**arXiv 2021**] A Taxonomy for Classification and Comparison of Dataflows for GNN Accelerators. *Garg et al.* [[Paper]](https://arxiv.org/abs/2103.07977)
* [**ISCAS 2021**] Characterizing the Communication Requirements of GNN Accelerators: A Model-Based Approach. *Guirado et al.* [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9401612)
* [**CAL 2020**] Characterizing and Understanding GCNs on GPU. *Yan et al.* [[Paper]](https://arxiv.org/abs/2010.00130)
* [**arXiv 2020**] Computing Graph Neural Networks: A Survey from Algorithms to Accelerators. *Abadal et al.* [[Paper]](https://arxiv.org/abs/2010.00130)
* [**KDD 2020**] Deep Graph Learning: Foundations, Advances and Applications. *Rong et al.* [[Paper]](https://dl.acm.org/doi/10.1145/3394486.3406474)
* [**TKDE   2020**] Deep Learning on Graphs: A Survey. *Zhang et al.*[[paper]](https://ieeexplore.ieee.org/abstract/document/9039675)
* [**arXiv  2021**] Graph Neural Networks: Methods, Applications, and Opportunities. *Waikhom et al.* [[Paper]](http://arxiv.org/abs/2108.10733)
* [**ISPASS  2021**] GNNMark: A Benchmark Suite to Characterize Graph Neural Network Training on GPUs. *Baruah et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9408205)
* [**CAL 2021**] Making a Better Use of Caches for GCN Accelerators with Feature Slicing and Automatic Tile Morphing.* [[Paper]](https://ieeexplore.ieee.org/document/9461595/)
* [**ISPASS  2021**] Performance Analysis of Graph Neural Network Frameworks. *Wu et al.* [[Paper]](https://www.semanticscholar.org/paper/Performance-Analysis-of-Graph-Neural-Network-Wu-Sun/b6da3ab0a6e710f16e11e5890818a107d1d5735c)
* [**arXiv 2021**] Sampling methods for efficient training of graph convolutional networks: A survey. *Liu et al.* [[Paper]](https://arxiv.org/abs/2103.05872)
* [**PPoPP 2021**] Understanding and bridging the gaps in current GNN performance optimizations. *Huang et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3437801.3441585)
* [**arXiv 2021**] Understanding GNN Computational Graph: A Coordinated Computation, IO, and Memory Perspective. *Zhang et al.* [[Paper]](http://arxiv.org/abs/2110.09524)
* [**arXiv 2021**] Understanding the Design Space of Sparse/Dense Multiphase Dataflows for Mapping Graph Neural Networks on Spatial Accelerators. *Garg et al.* [[Paper]](https://arxiv.org/pdf/2103.07977)


---
## Maintainers
- Ao Zhou, Beihang University. [[GitHub]](https://github.com/ZhouAo-ZA)
- Yingjie Qi, Beihang University. [[GitHub]](https://github.com/Kevin7Qi)
- Tong Qiao, Beihang University. [[GitHub]](https://github.com/qiaotonggg)



