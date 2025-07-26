# Literature on Graph Neural Networks Acceleration

A reading list for deep graph learning acceleration, including but not limited to related research on software and hardware level. The list covers related papers, conferences, tools, books, blogs, courses and other resources. We have a [team of Maintainers](#Maintainers) responsible for maintainance, meanwhile also welcome contributions from anyone.

Literatures in this page are arranged from a classification perspective, including the following topics:
- [Literature on Graph Neural Networks Acceleration](#literature-on-graph-neural-networks-acceleration)
  - [Hardware Acceleration for Graph Neural Networks](#hardware-acceleration-for-graph-neural-networks)
  - [System Designs for Deep Graph Learning](#system-designs-for-deep-graph-learning)
  - [Algorithmic Acceleration for Graph Neural Networks](#algorithmic-acceleration-for-graph-neural-networks)
  - [Surveys and Performance Analysis on Graph Learning](#surveys-and-performance-analysis-on-graph-learning)
  - [Maintainers](#maintainers)

You can also find [Related Conferences](./General%20Resources/Conference.md), [Graph Learning Tools](./General%20Resources/Frameworks%20%26%20Tools/), [Learning Materials on GNNs](./General%20Resources/Learning%20Materials) and Other Resources in [General Resources](./General%20Resources).



---
## Hardware Acceleration for Graph Neural Networks
* [**HPCA 2024**] MEGA: A Memory-Efficient GNN Accelerator Exploiting Degree-Aware Mixed-Precision Quantization.
  >*Zeyu Zhu, Fanrong Li, et al.* [[Paper]](https://arxiv.org/pdf/2311.09775.pdf)
* [**TC 2023**] Accelerating graph convolutional networks through a pim-accelerated approach.
  >*Jin H, Chen D, Zheng L, et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/10070790/)
* [**TPDS 2023**] GraphAGILE: An FPGA-Based Overlay Accelerator for Low-Latency GNN Inference.
  >*Zhang, Bingyi, Hanqing Zeng, and Viktor Prasanna.* [[paper]](https://ieeexplore.ieee.org/abstract/document/10158054)
* [**TC 2023**] Accelerating GNN Training by Adapting Large Graphs to Distributed Heterogeneous Architectures.
  >*Lizhi Zhang, Dongsheng Li, et al.* [[paper]](https://ieeexplore.ieee.org/abstract/document/10217071)
* [**CASES 2023**] MaGNAS: A Mapping-Aware Graph Neural Architecture Search Framework for Heterogeneous MPSoC Deployment.
  >*M Odema, H Bouzidi, H Ouarnoughi, et al.* [[paper]](https://arxiv.org/pdf/2307.08065)
* [**DAC 2023**] Lift: Exploiting Hybrid Stacked Memory for Energy-Efficient Processing of Graph Convolutional Networks.
  >*Jiaxian Chen, Zhaoyu Zhong, et al.* [[paper]](https://csse.szu.edu.cn/attachment/cglr/1681802011_Lift.pdf)
* [**DAC 2023**] ReRAM-based Graph Attention Network with Node-Centric Edge Searching and Hamming Similarity.
  >*Ruibin Mao, Xia Sheng, et al.* [[Paper]](http://)
* [**arXiv 2023**] HitGNN: High-throughput GNN Training Framework on CPU+Multi-FPGA Heterogeneous Platform.
  >*Lin Yichien, Prasanna Vikor, et al.* [[Paper]](http://arxiv.org/abs/2303.00158)
* [**TCAD 2023**] CoGNN: An Algorithm-Hardware Co-Design Approach to Accelerate GNN Inference With Minibatch Sampling.
  >*Zhong, Kai, Shulin Zeng, Wentao Hou, et al.* [[paper]](https://ieeexplore.ieee.org/abstract/document/10131995)
* [**TCAD 2023**] Algorithm/Hardware Co-optimization for Sparsity-Aware SpMM Acceleration of GNNs. 
  >*Gao Y, Gong L, Wang C, et al.* [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10139803)
* [**TCAD 2023**] CoGNN: An Algorithm-Hardware Co-Design Approach to Accelerate GNN Inference with Mini-Batch Sampling. 
  >*Zhong K, Zeng S, Hou W, et al.* [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10131995)
* [**GSVLSI 2023**] IMA-GNN: In-Memory Acceleration of Centralized and Decentralized Graph Neural Networks at the Edge. 
  >*Morsali M, Nazzal M, Khreishah A, et al.* [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3583781.3590248)
* [**JCST 2023**] GShuttle: Optimizing Memory Access Efficiency for Graph Convolutional Neural Network Accelerators. 
  >*Li J J, Wang K, Zheng H, et al..* [[Paper]](https://link.springer.com/article/10.1007/s11390-023-2875-9)
* [**arXiv 2023**] Dynasparse: Accelerating GNN Inference through Dynamic Sparsity Exploitation. 
  >*Bingyi Zhang, Viktor Prasanna.* [[Paper]](https://arxiv.org/pdf/2303.12901)
* [**arXiv 2023**] GNNBuilder: An Automated Framework for Generic Graph Neural Network Accelerator Generation, Simulation, and Optimization. 
  >*Abi-Karam S, Hao C.* [[Paper]](https://arxiv.org/pdf/2303.16459)
* [**arXiv 2023**] HitGNN: High-throughput GNN Training Framework on CPU+Multi-FPGA Heterogeneous Platform. 
  >*Lin Y C, Zhang B.* [[Paper]](https://arxiv.org/pdf/2303.01568)
* [**ICCD 2022**] CoDG-ReRAM: An Algorithm-Hardware Co-design to Accelerate Semi-Structured GNNs on ReRAM. 
  >*Luo Y, Behnam P, Thorat K, et al.* [[Paper]](https://khan.engr.uconn.edu/pubs/iccd22-reram-gnn.pdfO)
* [**HPCA 2022**] Accelerating Graph Convolutional Networks Using Crossbar-based Processing-In-Memory Architectures. 
  >*Huang Y, Zheng L, Yao P, et al.* [[Paper]](https://www.computer.org/csdl/proceedings-article/hpca/2022/202700b029/1Ds0gRvUFjO)
* [**HPCA 2022**] GCoD: Graph Convolutional Network Acceleration via Dedicated Algorithm and Accelerator Co-Design. 
  >*You H, Geng T, Zhang Y, et al.* [[Paper]](https://arxiv.org/pdf/2112.11594) [[GitHub]](https://github.com/rice-eic/gcod)
* [**HPCA 2022**] ReGNN: A Redundancy-Eliminated Graph Neural Networks Accelerator.
  >*Chen C, Li K, Li Y, et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9773273)
* [**ISCA 2022**] DIMMining: pruning-efficient and parallel graph mining on near-memory-computing. 
  >*Dai G, Zhu Z, Fu T, et al.* [[Paper]](https://doi.org/10.1145/3470496.3527388)
* [**ISCA 2022**] Hyperscale FPGA-as-a-service architecture for large-scale distributed graph neural network.
  >*Li S, Niu D, Wang Y, et al.* [[Paper]](https://dl.acm.org/doi/10.1145/3470496.3527439)
* [**DAC 2022**] Improving GNN-Based Accelerator Design Automation with Meta Learning. 
  >*Bai Y, Sohrabizadeh A, Sun Y, et al.* [[Paper]](https://vast.cs.ucla.edu/sites/default/files/publications/_DAC_22__GNN_DSE_MAML.pdf)
* [**CICC 2022**] StreamGCN: Accelerating Graph Convolutional Networks with Streaming Processing.
  >*Sohrabizadeh A, Chi Y, Cong J.* [[Paper]](https://web.cs.ucla.edu/~atefehsz/publication/StreamGCN-CICC22.pdf)
* [**IPDPS 2022**] Model-Architecture Co-Design for High Performance Temporal GNN Inference on FPGA.
  >*Zhou H, Zhang B, Kannan R, et al.* [[Paper]](https://arxiv.org/abs/2203.05095)
* [**TPDS 2022**] SGCNAX: A Scalable Graph Convolutional Neural Network Accelerator With Workload Balancing. 
  >*Li J, Zheng H, Wang K, et al.* [[Paper]](https://www.computer.org/csdl/journal/td/2022/11/09645224/1zc6JTLADC0)
* [**TCSI 2022**] A Low-Power Graph Convolutional Network Processor With Sparse Grouping for 3D Point Cloud Semantic Segmentation in Mobile Devices. 
  >*Kim S, Kim S, Lee J, et al.* [[Paper]](https://ieeexplore.ieee.org/document/9669025)
* [**JAHC 2022**] DRGN: a dynamically reconfigurable accelerator for graph neural networks.
  >*Yang C, Huo K B, Geng L F, et al.* [[Paper]](https://link.springer.com/article/10.1007/s12652-022-04402-x)
* [**JSA 2022**] Algorithms and architecture support of degree-based quantization for graph neural networks.
  >*Guo Y, Chen Y, Zou X, et al.* [[Paper]](https://dl.acm.org/doi/10.1016/j.sysarc.2022.102578)
* [**JSA 2022**] QEGCN: An FPGA-based accelerator for quantized GCNs with edge-level parallelism.
  >*Yuan W, Tian T, Wu Q, et al.* [[Paper]](https://dl.acm.org/doi/10.1016/j.sysarc.2022.102596)
* [**FCCM 2022**] GenGNN: A Generic FPGA Framework for Graph Neural Network Acceleration. 
  >*Abi-Karam S, He Y, Sarkar R, et al.* [[Paper]]( https://arxiv.org/pdf/2201.08475) [[GitHub]](https://github.com/sharc-lab/gengnn)
* [**FAST 2022**] Hardware/Software Co-Programmable Framework for Computational SSDs to Accelerate Deep Learning Service on Large-Scale Graphs. 
  >*Kwon M, Gouk D, Lee S, et al.* [[Paper]](https://arxiv.org/abs/2201.09189)
* [**arXiv 2022**] DFG-NAS: Deep and Flexible Graph Neural Architecture Search.
  >*Zhang W, Lin Z, Shen Y, et al.* [[Paper]](https://arxiv.org/abs/2206.08582)
* [**arXiv 2022**] GROW: A Row-Stationary Sparse-Dense GEMM Accelerator for Memory-Efficient Graph Convolutional Neural Networks. 
  >*Kang M, Hwang R, Lee J, et al.* [[Paper]](https://arxiv.org/abs/2203.00158)
* [**arXiv 2022**] Enabling Flexibility for Sparse Tensor Acceleration via Heterogeneity. 
  >*Qin E, Garg R, Bambhaniya A, et al.* [[Paper]](https://arxiv.org/abs/2201.08916)
* [**arXiv 2022**] FlowGNN: A Dataflow Architecture for Universal Graph Neural Network Inference via Multi-Queue Streaming. 
  >*Sarkar R, Abi-Karam S, He Y, et al.* [[Paper]](https://arxiv.org/abs/2204.13103) [[GitHub]](https://github.com/sharc-lab/flowgnn)
* [**arXiv 2022**] Low-latency Mini-batch GNN Inference on CPU-FPGA Heterogeneous Platform. 
  >*Zhang B, Zeng H, Prasanna V.* [[Paper]](http://arxiv.org/abs/2206.08536)
* [**arXiv 2022**] SmartSAGE: Training Large-scale Graph Neural Networks using In-Storage Processing Architectures. 
  >*Lee Y, Chung J, Rhu M.* [[Paper]](http://arxiv.org/abs/2205.04711)
* [**MICRO 2021**] AWB-GCN: A Graph Convolutional Network Accelerator with Runtime Workload Rebalancing. 
  >*Geng T, Li A, Shi R, et al.* [[Paper]](https://ieeexplore.ieee.org/document/9252000)
* [**MICRO 2021**] Point-X: A Spatial-Locality-Aware Architecture for Energy-Efficient Graph-Based Point-Cloud Deep Learning. 
  >*Zhang J F, Zhang Z.*  [[Paper]](https://dl.acm.org/doi/10.1145/3466752.3480081)
* [**HPCA 2021**] GCNAX: A Flexible and Energy-efficient Accelerator for Graph Convolutional Neural Networks. 
  >*Li J, Louri A, Karanth A, et al.* [[Paper]](https://doi.org/10.1109/HPCA51647.2021.00070)
* [**DAC 2021**] DyGNN: Algorithm and Architecture Support of Dynamic Pruning for Graph Neural Networks. 
  >*Chen C, Li K, Zou X, et al.* [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9586298)
* [**DAC 2021**] BlockGNN: Towards Efficient GNN Acceleration Using Block-Circulant Weight Matrices. 
  >*Zhou Z, Shi B, Zhang Z, et al.* [[Paper]](https://arxiv.org/abs/2104.06214)
* [**DAC 2021**] GNNerator: A Hardware/Software Framework for Accelerating Graph Neural Networks. 
  >*Stevens J R, Das D, Avancha S, et al.* [[Paper]](https://arxiv.org/abs/2103.10836)
* [**DAC 2021**] PIMGCN: A ReRAM-Based PIM Design for Graph Convolutional Network Acceleration. 
  >*Yang T, Li D, Han Y, et al.*  [[Paper]](https://ieeexplore.ieee.org/document/9586231)
* [**TCAD 2021**] Rubik: A Hierarchical Architecture for Efficient Graph Neural Network Training. 
  >*Chen X, Wang Y, Xie X, et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9428002)
* [**TCAD 2021**] Cambricon-G: A Polyvalent Energy-efficient Accelerator for Dynamic Graph Neural Networks. 
  >*Song X, Zhi T, Fan Z, et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9326339)
* [**ICCAD 2021**] DARe: DropLayer-Aware Manycore ReRAM architecture for Training Graph Neural Networks. 
  >*Arka A I, Joardar B K, Doppa J R, et al.* [[Paper]](https://ieeexplore.ieee.org/document/9643511)
* [**DATE 2021**] ReGraphX: NoC-Enabled 3D Heterogeneous ReRAM Architecture for Training Graph Neural Networks. 
  >*Arka A I, Doppa J R, Pande P P, et al.* [[Paper]](https://doi.org/10.23919/DATE51398.2021.9473949)
* [**FCCM 2021**] BoostGCN: A Framework for Optimizing GCN Inference on FPGA. 
  >*Zhang B, Kannan R, Prasanna V.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9444065)
* [**SCIS 2021**] Towards efficient allocation of graph convolutional networks on hybrid computation-in-memory architecture.
  >*Chen J, Lin G, Chen J, et al.* [[Paper]](https://link.springer.com/article/10.1007/s11432-020-3248-y)
* [**EuroSys 2021**] Tesseract: distributed, general graph pattern mining on evolving graphs. 
  >*Bindschaedler L, Malicevic J, Lepers B, et al.* [[Paper]](https://doi.org/10.1145/3447786.3456253)
* [**EuroSys 2021**] Accelerating Graph Sampling for Graph Machine Learning Using GPUs. 
  >*Jangda A, Polisetty S, Guha A, et al.* [[Paper]]( https://doi.org/10.1145/3447786.3456244)
* [**ATC 2021**] GLIST: Towards In-Storage Graph Learning. 
  >*Li C, Wang Y, Liu C, et al.* [[Paper]](www.usenix.org/conference/atc21/presentation/li-cangyuan)
* [**CAL 2021**] Hardware Acceleration for GCNs via Bidirectional Fusion. 
  >*Li H, Yan M, Yang X, et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9425440)
* [**arXiv 2021**] GNNIE: GNN Inference Engine with Load-balancing and Graph-Specific Caching. 
  >*Mondal S, Manasi S D, Kunal K, et al.* [[Paper]](http://arxiv.org/abs/2105.10554)
* [**arXiv 2021**] LW-GCN: A Lightweight FPGA-based Graph Convolutional Network Accelerator. 
  >*Tao Z, Wu C, Liang Y, et al.*  [[Paper]](https://arxiv.org/abs/2111.03184)
* [**arXiv 2021**] VersaGNN: a Versatile accelerator for Graph neural networks. 
  >*Shi F, Jin A Y, Zhu S C.* [[Paper]](https://arxiv.org/abs/2105.01280)
* [**arXiv 2021**] ZIPPER: Exploiting Tile- and Operator-level Parallelism for General and Scalable Graph Neural Network Acceleration. 
  >*Zhang Z, Leng J, Lu S, et al.* [[Paper]](https://arxiv.org/abs/2107.08709v1)
* [**HPCA 2020**] HyGCN: A GCN Accelerator with Hybrid Architecture. 
  >*Yan M, Deng L, Hu X, et al.* [[Paper]](https://doi.org/10.1109/HPCA47549.2020.00012)
* [**DAC 2020**] Hardware Acceleration of Graph Neural Networks. 
  >*Auten A, Tomei M, Kumar R.* [[Paper]](https://doi.org/10.1109/dac18072.2020.9218751)
* [**ICCAD 2020**] DeepBurning-GL: an automated framework for generating graph neural network accelerators. 
  >*Liang S, Liu C, Wang Y, et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9256539)
* [**TC 2020**] EnGN: A High-Throughput and Energy-Efficient Accelerator for Large Graph Neural Networks. 
  >*Liang S, Wang Y, Liu C, et al.* [[Paper]](https://ieeexplore.ieee.org/document/9161360/)
* [**SC 2020**] GE-SpMM: General-Purpose Sparse Matrix-Matrix Multiplication on GPUs for Graph Neural Networks. 
  >*Huang G, Dai G, Wang Y, et al.* [[Paper]](https://doi.org/10.1109/sc41405.2020.00076) [[GitHub]](https://github.com/hgyhungry/ge-spmm)
* [**CCIS 2020**] GNN-PIM: A Processing-in-Memory Architecture for Graph Neural Networks. 
  >*Wang Z, Guan Y, Sun G, et al.* [[Paper]](https://www.semanticscholar.org/paper/GNN-PIM%3A-A-Processing-in-Memory-Architecture-for-Wang-Guan/1d03e4bebc9cf3c3fdd9204504d92b20d97d1fdf)
* [**FPGA 2020**] GraphACT: Accelerating GCN Training on CPU-FPGA Heterogeneous Platforms. 
  >*Zeng H, Prasanna V.* [[Paper]](https://arxiv.org/abs/2001.02498) [[GitHub]](https://github.com/GraphSAINT/GraphACT)
* [**ICPADS 2020**] S-GAT: Accelerating Graph Attention Networks Inference on FPGA Platform with Shift Operation. 
  >*Yan W, Tong W, Zhi X.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9359183)
* [**ASAP 2020**] Hardware Acceleration of Large Scale GCN Inference. 
  >*Zhang B, Zeng H, Prasanna V.*[[Paper]](https://ieeexplore.ieee.org/document/9153263)
* [**ICA3PP 2020**] Towards a Deep-Pipelined Architecture for Accelerating Deep GCN on a Multi-FPGA Platform. 
  >*Cheng Q, Wen M, Shen J, et al.* [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-60245-1_36)
* [**Access 2020**] FPGAN: An FPGA Accelerator for Graph Attention Networks With Software and Hardware Co-Optimization. 
  >*Yan W, Tong W, Zhi X.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9195849)
* [**arXiv 2020**] GRIP: A Graph Neural Network Accelerator Architecture. 
  >*Kiningham K, Re C, Levis P.* [[Paper]](https://arxiv.org/abs/2007.13828)
* [**ASICON 2019**] An FPGA Implementation of GCN with Sparse Adjacency Matrix. 
  >*Ding L, Huang Z, Chen G.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/8983647)


---
## System Designs for Deep Graph Learning
* [**ICS 2025**] YH-Light: Yielding Hierarchy-aware Partitioner for Large-scale Graph Processing.
  >*Gan X, Li T, Gong C, et al.* [[Paper]](https://hpcrl.github.io/ICS2025-webpage/program/Proceedings_ICS25/ics25-1.pdf)
* [**TRETS 2025**]DGMF:aunified dynamic mapping framework for graph neural networks.
  >*Fan Z, Peng M.* [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3744345)
* [**DAC 2024**] GNNavigator: Towards Adaptive Training of Graph Neural Networks via Automatic Guideline Exploration.
  >*Tong Qiao, Jianlei Yang, et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3649329.3656504)
* [**DAC 2024**] Graph neural networks automated design and deployment on device-edge co-inference systems.
  >*Ao Zhou, Jianlei Yang, et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3649329.3655938)
* [**HPCA 2024**] MEGA: A Memory-Efficient GNN Accelerator Exploiting Degree-Aware Mixed-Precision Quantization.
  >*Zhu Z, Li F, Li G, et al.* [[paper]]((https://arxiv.org/pdf/2405.04114)
* [**WWW 2024**] λGrapher: A Resource-Efficient Serverless System for GNN Serving through Graph Sharing.
  >*Hu H, Liu F, Pei Q, et al.* [[paper]](https://fangmingliu.github.io/files/WWW24-LambdaGrapher.pdf)
* [**IPDPS 2024**] ARGO: An Auto-Tuning Runtime System for Scalable GNN Training on Multi-Core Processor.
  >*Yi-Chien Lin, Yuyang Chen, et al.* [[paper]](https://arxiv.org/abs/2402.03671)
* [**SC 2023**] BLAD: Adaptive Load Balanced Scheduling and Operator Overlap Pipeline For Accelerating The Dynamic GNN Training.
  >*Fu, Kaihua, et al.* [[paper]](https://dl.acm.org/doi/abs/10.1145/3581784.3607040)
* [**SOSP 2023**] gSampler: General and Efficient GPU-based Graph Sampling for Graph Learning.
  >*Gong, Ping, et al.* [[paper]](https://dl.acm.org/doi/abs/10.1145/3600006.3613168)
* [**ICDE 2023**] InferTurbo: A Scalable System for Boosting Full-graph Inference of Graph Neural Network over Huge Graphs.
  >*Zhang, Dalong, et al.* [[paper]](https://ieeexplore.ieee.org/abstract/document/10184616/)
* [**IPDPS 2023**] Communication Optimization for Distributed Execution of Graph Neural Networks.
  >*Kurt, Süreyya Emre, et al.* [[paper]](https://ieeexplore.ieee.org/abstract/document/10177386/)
* [**IPDPS 2023**] Betty: Enabling Large-Scale GNN Training with Batch-Level Graph Partitioning.
  >*Yang, Shuangyan, et al.* [[paper]](https://dl.acm.org/doi/abs/10.1145/3575693.3575725)
* [**LOG 2023**] PyTorch-Geometric Edge - a Library for Learning Representations of Graph Edges.
  >*Bielak, Piotr, and Tomasz Jan Kajdanowicz.* [[paper]](https://openreview.net/forum?id=hM5UIWqZ7d)
* [**LOG 2023**] FreshGNN: Reducing Memory Access via Stable Historical Embeddings for Graph Neural Network Training.
  >*Kezhao Huang, Haitian Jiang, Haitian_Jiang3, et al.* [[paper]](https://dl.acm.org/doi/abs/10.1145/3572848.3577487)
* [**PPoPP 2023**] PiPAD: Pipelined and Parallel Dynamic GNN Training on GPUs.
  >*Wang, Chunyang, Desen Sun, and Yuebin Bai.* [[paper]](https://dl.acm.org/doi/abs/10.1145/3572848.3577487)
* [**PPoPP 2023**] DSP: Efficient GNN Training with Multiple GPUs.
  >*Cai Z, Zhou Q, et al.* [[Paper]](https://assets.amazon.science/70/b4/c4efd54f44b695d62dd7f441fccb/dsp-efficient-gnn-training-with-multiple-gpus.pdf)
* [**ICS 2023**] BitGNN: Unleashing the Performance Potential of Binary Graph Neural Networks on GPUs.
  >*Chen, Jou-An, Hsin-Hsuan Sung, Xipeng Shen, et al.* [[paper]](https://dl.acm.org/doi/abs/10.1145/3577193.3593725)
* [**JSAC 2022**] GNN at the Edge: Cost-Efficient Graph Neural Network Processing Over Distributed Edge Servers.
  >*Zeng, Liekang, Chongyu Yang, Peng Huang, et al.* [[paper]](https://ieeexplore.ieee.org/abstract/document/9996395)
* [**TC 2023**] TurboGNN: Improving the End-to-End Performance for Sampling-Based GNN Training on GPUs.
  >*Wenchao Wu, Xuanhua Shi, et al.* [[Paper]](https://ieeexplore.ieee.org/document/10070878)
* [**TPDS 2023**] TurboMGNN: Improving Concurrent GNN Training Tasks on GPU With Fine-Grained Kernel Fusion.
  >*Wenchao Wu, Xuanhua Shi, et al.* [[Paper](https://ieeexplore.ieee.org/document/10103627)
* [**TPDPS 2023**] HyScale-GNN: A Scalable Hybrid GNN Training System on Single-Node Heterogeneous Architecture.
  >*Lin Yichien, Prasanna Vikor, et al.* [[Paper]](http://arxiv.org/abs/2303.00158)
* [**INFOCOM 2023**] Two-level Graph Caching for Expediting Distributed GNN Training.
  >*Ziyue Luo et al.* [[Paper]](https://i.cs.hku.hk/~cwu/papers/zzhang-infocom23.pdf)
* [**NSDI 2023**] BGL: GPU-Efficient GNN Training by Optimizing Graph Data I/O and Preprocessing.
  >*Liu T, Chen Y, Li D, et al.* [[Paper]](https://arxiv.org/abs/2112.08541)
* [**arXiv 2023**] GraphTensor: Comprehensive GNN-Acceleration Framework for Efficient Parallel Processing of Massive Datasets.
  >*Jang J, Kwon M, Gouk D, et al.* [[Paper]](https://arxiv.org/pdf/2305.17469)
* [**arXiv 2022**] DistGNN-MB: Distributed Large-Scale Graph Neural Network Training on x86 via Minibatch Sampling.
  >*Vasimuddin M, Mohanty R, Misra S, et al.* [[Paper]](https://arxiv.org/abs/2211.06385)
* [**VLDB 2022**] ByteGNN: efficient graph neural network training at large scale. 
  >*Zheng C, Chen H, Cheng Y, et al.* [[Paper]](https://dl.acm.org/doi/abs/10.14778/3514061.3514069)
* [**EuroSys 2022**] GNNLab: a factored system for sample-based GNN training over GPUs. 
  >*Yang J, Tang D, Song X, et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3492321.3519557)
* [**PPoPP 2022**] Rethinking graph data placement for graph neural network training on multiple GPUs.
  >*Song S, Jiang P.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3503221.3508435)
* [**TC 2022**] Multi-node Acceleration for Large-scale GCNs. 
  >*Sun, Gongjian, et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9893364/)
* [**ISCA 2022**] Graphite: optimizing graph neural networks on CPUs through cooperative software-hardware techniques. 
  >*Gong Z, Ji H, Yao Y, et al.* [[Paper]](https://dl.acm.org/doi/10.1145/3470496.3527403)
* [**PPoPP 2022**] QGTC: accelerating quantized graph neural networks via GPU tensor core.
  >*Wang Y, Feng B, Ding Y.* [[Paper]](https://dl.acm.org/doi/10.1145/3503221.3508408)
* [**SIGMOD 2022**] NeutronStar: Distributed GNN Training with Hybrid Dependency Management. 
  >*Wang Q, Zhang Y, Wang H, et al.* [[Paper]](https://doi.org/10.1145/3514221.3526134)
* [**MLSys 2022**] Accelerating Training and Inference of Graph Neural Networks with Fast Sampling and Pipelining. 
  >*Kaler T, Stathas N, Ouyang A, et al.* [[Paper]](https://proceedings.mlsys.org/paper/2022/hash/35f4a8d465e6e1edc05f3d8ab658c551-Abstract.html)
* [**KDD 2022**] Distributed Hybrid CPU and GPU training for Graph Neural Networks on Billion-Scale Heterogeneous Graphs.
  >*Zheng D, Song X, Yang C, et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3534678.3539177)
* [**FPGA 2022**] SPA-GCN: Efficient and Flexible GCN Accelerator with Application for Graph Similarity Computation. 
  >*Sohrabizadeh A, Chi Y, Cong J.* [[Paper]]( https://arxiv.org/pdf/2111.05936)
* [**HPDC 2022**] TLPGNN: A Lightweight Two-Level Parallelism Paradigm for Graph Neural Network Computation on GPU. 
  >*Fu Q, Ji Y, Huang H H.* [[Paper]](https://doi.org/10.1145/3502181.3531467)
* [**Concurrency and Computation 2022**] BRGraph: An efficient graph neural network training system by reusing batch data on GPU.
  >*Ge K, Ran Z, Lai Z, et al.* [[Paper]](https://onlinelibrary.wiley.com/doi/10.1002/cpe.6961)
* [**arXiv 2022**] Improved Aggregating and Accelerating Training Methods for Spatial Graph Neural Networks on Fraud Detection.
  >*Zeng Y, Tang J.* [[Paper]](http://arxiv.org/abs/2202.06580)
* [**arXiv 2022**] Marius++: Large-scale training of graph neural networks on a single machine. 
  >*Waleffe R, Mohoney J, Rekatsinas T, et al.* [[Paper]](https://arxiv.org/abs/2202.02365)
* [**HPCA 2021**] DistGNN: scalable distributed training for large-scale graph neural networks.
  >*Md V, Misra S, Ma G, et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3458817.3480856)
* [**CLUSTER 2021**] 2PGraph: Accelerating GNN Training over Large Graphs on GPU Clusters.
  >*Zhang L, Lai Z, Li S, et al.* [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9556026)
* [**APSys 2021**] Accelerating GNN training with locality-aware partial execution.
  >*Kim T, Hwang C, Park K S, et al. * [[Paper]](https://dl.acm.org/doi/abs/10.1145/3476886.3477515)
* [**JPDC  2021**] Accurate, efficient and scalable training of Graph Neural Networks. 
  >*Zeng H, Zhou H, Srivastava A, et al.* [[Paper]](https://www.sciencedirect.com/science/article/pii/S0743731520303579) [[GitHub]](https://github.com/GraphSAINT/GraphSAINT)
* [**JPDC  2021**] High performance GPU primitives for graph-tensor learning operations. 
  >*Zhang T, Kan W, Liu X Y.* [[Paper]](https://www.sciencedirect.com/science/article/abs/pii/S0743731520304007)
* [**OSDI  2021**] Dorylus: Affordable, Scalable, and Accurate GNN Training with Distributed CPU Servers and Serverless Threads. 
  >*Thorpe J, Qiao Y, Eyolfson J, et al.* [[Paper]](https://arxiv.org/abs/2105.11118) [[GitHub]](https://github.com/uclasystem/dorylus)
* [**OSDI   2021**] GNNAdvisor: An Adaptive and Efﬁcient Runtime System for GNN Acceleration on GPUs 
  >*Wang Y, Feng B, Li G, et al.* [[Paper]](https://www.usenix.org/system/files/osdi21-wang-yuke.pdf) [[GitHub]](https://github.com/YukeWang96/OSDI21_AE)
* [**EuroSys 2021**] DGCL: an efficient communication library for distributed GNN training. 
  >*Cai Z, Yan X, Wu Y, et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3447786.3456233)
* [**EuroSys  2021**] FlexGraph: a flexible and efficient distributed framework for GNN training. 
  >*Wang L, Yin Q, Tian C, et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3447786.3456229) [[GitHub]](https://github.com/snudatalab/FlexGraph)
* [**EuroSys 2021**] Seastar: vertex-centric programming for graph neural networks. 
  >*Wu Y, Ma K, Cai Z, et al.* [[Paper]](https://doi.org/10.1145/3447786.3456247)
* [**TPDS  2021**] Efficient Data Loader for Fast Sampling-Based GNN Training on Large Graphs. 
  >*Bai Y, Li C, Lin Z, et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9376972)
* [**GNNSys 2021**] FedGraphNN: A Federated Learning System and Benchmark for Graph Neural Networks.
  >*He C, Balasubramanian K, Ceyani E, et al.* [[Paper]](https://gnnsys.github.io/papers/GNNSys21_paper_3.pdf) [[Poster]](https://gnnsys.github.io/posters/GNNSys21_poster_3.pdf) [[GitHub]](https://github.com/FedML-AI/FedGraphNN)
* [**GNNSys 2021**] Graphiler: A Compiler for Graph Neural Networks.
  >*Xie Z, Ye Z, Wang M, et al.* [[Paper]](https://gnnsys.github.io/papers/GNNSys21_paper_10.pdf) [[Poster]](https://gnnsys.github.io/posters/GNNSys21_poster_10.pdf)
* [**GNNSys 2021**] IGNNITION: A framework for fast prototyping of Graph Neural Networks.
  >*Pujol Perich D, Suárez-Varela Maciá J R, Ferriol Galmés M, et al.* [[Paper]](https://gnnsys.github.io/papers/GNNSys21_paper_4.pdf) [[Poster]](https://gnnsys.github.io/posters/GNNSys21_poster_4.pdf)
* [**GNNSys 2021**] Load Balancing for Parallel GNN Training.
  >*Su Q, Wang M, Zheng D, et al* [[Paper]](https://gnnsys.github.io/papers/GNNSys21_paper_18.pdf) [[Poster]](https://gnnsys.github.io/posters/GNNSys21_poster_18.pdf)
* [**IPDPS  2021**] FusedMM: A Unified SDDMM-SpMM Kernel for Graph Embedding and Graph Neural Networks. 
  >*Rahman M K, Sujon M H, Azad A.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9460486) [[GitHub]](https://github.com/HipGraph/FusedMM)
* [**IPCCC 2021**] Accelerate graph neural network training by reusing batch data on GPUs.
  >*Ran Z, Lai Z, Zhang L, et al. * [[Paper]](https://ieeexplore.ieee.org/abstract/document/9679362/)
* [**arXiv  2021**] PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models. 
  >*Rozemberczki B, Scherer P, He Y, et al.* [[Paper]](https://arxiv.org/abs/2104.07788) [[GitHub]](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)
* [**arXiv 2021**] QGTC: Accelerating Quantized GNN via GPU Tensor Core. 
  >*Wang Y, Feng B, Ding Y.* [[Paper]](http://arxiv.org/abs/2111.09547) [[GitHub]](https://github.com/YukeWang96/PPoPP22_QGTC)
* [**arXiv 2021**] TC-GNN: Accelerating Sparse Graph Neural Network Computation Via Dense Tensor Core on GPUs. 
  >*Wang Y, Feng B, Ding Y.* [[Paper]](http://arxiv.org/abs/2112.02052) [[GitHub]](https://github.com/YukeWang96/TCGNN-Pytorch)
* [**ICCAD 2020**] fuseGNN: accelerating graph convolutional neural network training on GPGPU. 
  >*Chen Z, Yan M, Zhu M, et al.* [[Paper]](https://ieeexplore.ieee.org/document/9256702) [[GitHub]](https://github.com/apuaaChen/gcnLib)
* [**VLDB 2020**] AGL: a scalable system for industrial-purpose graph machine learning.
  >*Zhang D, Huang X, Liu Z, et al.* [[Paper]](https://arxiv.org/pdf/2003.02454)
* [**SC 2020**] FeatGraph: A Flexible and Efficient Backend for Graph Neural Network Systems. 
  >*Hu Y, Ye Z, Wang M, et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9355318) [[Github]](https://github.com/amazon-research/FeatGraph)
* [**MLSys  2020**] Improving the Accuracy, Scalability, and Performance of  Graph Neural Networks with Roc. 
  >*Jia Z, Lin S, Gao M, et al.* [[Paper]](https://www-cs.stanford.edu/people/matei/papers/2020/mlsys_roc.pdf)
* [**CVPR 2020**] L2-GCN: Layer-Wise and Learned Efficient Training of Graph Convolutional Networks.
  >*You Y, Chen T, Wang Z, et al.* [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/You_L2-GCN_Layer-Wise_and_Learned_Efficient_Training_of_Graph_Convolutional_Networks_CVPR_2020_paper.html) 
* [**TPDS  2020**] EDGES: An Efficient Distributed Graph Embedding System on GPU Clusters. 
  >*Yang D, Liu J, Lai J.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9272876)
* [**AccML  2020**] GIN : High-Performance, Scalable Inference for Graph Neural Networks. 
  >*Fu Q, Huang H H.* [[Paper]](https://workshops.inf.ed.ac.uk/accml/papers/2020/AccML_2020_paper_6.pdf)
* [**SoCC  2020**] PaGraph: Scaling GNN training on large graphs via computation-aware caching. 
  >*Lin Z, Li C, Miao Y, et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3419111.3421281)
* [**IPDPS  2020**] Pcgcn: Partition-centric processing for accelerating graph convolutional network. 
  >*Tian C, Ma L, Yang Z, et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9139807/)
* [**arXiv 2020**] Deep graph library optimizations for intel (r) x86 architecture.
  >*Avancha S, Md V, Misra S, et al.* [[Paper]](https://arxiv.org/abs/2007.06354)
* [**IA3  2020**] DistDGL: Distributed Graph Neural Network Training for Billion-Scale Graphs. 
  >*Zheng D, Ma C, Wang M, et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9407264)
* [**CoRR  2019**] Deep Graph Library: Towards Efficient and Scalable Deep Learning on Graphs. 
  >*Wang M Y.* [[Paper]](https://arxiv.org/abs/1909.01315v2) [[GitHub]](https://github.com/dmlc/dgl/) [[Home Page]](https://www.dgl.ai/)
* [**ICLR  2019**] Fast Graph Representation Learning with PyTorch Geometric. 
  >*Fey M, Lenssen J E.* [[Paper]](https://arxiv.org/abs/1903.02428) [[GitHub]](https://github.com/rusty1s/pytorch_geometric) [[Documentation]](https://pytorch-geometric.readthedocs.io/en/latest/)
* [**KDD  2019**] AliGraph: a comprehensive graph neural network platform. 
  >*Yang H.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3292500.3340404) [[GitHub]](https://github.com/alibaba/graph-learn)
* [**SysML 2019**] PyTorch-BigGraph: A Large-scale Graph Embedding System. 
  >*Lerer A, Wu L, Shen J, et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3292500.3340404) [[GitHub]](https://github.com/facebookresearch/PyTorch-BigGraph)
* [**ATC  2019**] NeuGraph: Parallel Deep Neural Network Computation on Large Graphs. 
  >*Ma L, Yang Z, Miao Y, et al.* [[Paper]](https://www.usenix.org/conference/atc19/presentation/ma)
* [**arXiv 2018**] Relational inductive biases, deep learning, and graph networks. 
  >*Battaglia P W, Hamrick J B, Bapst V, et al.* [[Paper]](https://arxiv.org/abs/1806.01261) [[GitHub]](https://github.com/deepmind/graph_nets)
---
## Algorithmic Acceleration for Graph Neural Networks
* [**TMLR 2025**] Sparse Decomposition of Graph Neural Networks.
  >*Yaochen Hu, Mai Zeng, et al.* [[Paper]](https://openreview.net/forum?id=xdWP1d8BxI)
* [**TC 2024**] HGNAS: Hardware-Aware Graph Neural Architecture Search for Edge Devices.
  >*Ao Zhou, Jianlei Yang, et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/10644077)
* [**TC 2023**] Approximation- and Quantization-Aware Training for Graph Neural Networks.
  >*Novkin R, Klemme F, Amrouch H* [[Paper]](https://ieeexplore.ieee.org/abstract/document/10335640)
* [**TC 2023**] Sugar: Efficient subgraph-level training via resource-aware graph partitioning.
  >*Xue Z, Yang Y, Marculescu R* [[Paper]](https://ieeexplore.ieee.org/abstract/document/10168538/)
* [**ASC 2023**] Imbalanced node classification with Graph Neural Networks: A unified approach leveraging homophily and label information.
  >*Lv D, Xu Z, Zhang J, et al.*  [[Paper]](https://www.sciencedirect.com/science/article/abs/pii/S1568494623010037)
* [**arXiv 2023**] AdaptGear: Accelerating GNN Training via Adaptive Subgraph-Level Kernels on GPUs.
  >*Yangjie Zhou, Jingwen Leng, et al.*  [[Paper]](https://subjectnoi.github.io/about/AdaptGear.pdf)
* [**DAC 2023**] Hardware-Aware Graph Neural Network Automated Design for Edge Computing Platforms.
  >*Ao Zhou, Jianlei Yang, et al.* [[Paper]](https://arxiv.org/pdf/2303.10875)
* [**arXiv 2023**] Provably Convergent Subgraph-wise Sampling for Fast GNN Training.
  >*Wang Jie, Shi Zhihao, et al..* [[Paper]](https://arxiv.org/abs/2303.11081)
* [**arXiv 2023**] LazyGNN: Large-Scale Graph Neural Networks via Lazy Propagation.
  >*Rui Xue, Haoyu Han, et al..* [[Paper]](https://arxiv.org/abs/2302.01503)
* [**TPDS 2022**] Accelerating Backward Aggregation in GCN Training With Execution Path Preparing on GPUs.
  >*Shaoxian Xu, Zhiyuan Shao, et al.* [[Paper]](https://ieeexplore.ieee.org/document/9887890) 
* [**AAAI 2022**] Early-Bird GCNs: Graph-Network Co-Optimization Towards More Efficient GCN Training and Inference via Drawing Early-Bird Lottery Tickets. 
  >*You H, Lu Z, Zhou Z, et al.* [[Paper]](https://www.researchgate.net/profile/Haoran-You/publication/349704520_GEBT_Drawing_Early-Bird_Tickets_in_Graph_Convolutional_Network_Training/links/61e0930dc5e3103375916c9f/GEBT-Drawing-Early-Bird-Tickets-in-Graph-Convolutional-Network-Training.pdf) [[GitHub]](https://github.com/RICE-EIC/Early-Bird-GCN)
* [**ICLR 2022**] Adaptive Filters for Low-Latency and Memory-Efficient Graph Neural Networks. 
  >*Tailor S A, Opolka F, Lio P, et al.* [[Paper]](https://openreview.net/forum?id=hl9ePdHO4_s) [[GitHub]](https://github.com/shyam196/egc)
* [**ICLR 2022**] Graph-less Neural Networks: Teaching Old MLPs New Tricks Via Distillation. 
  >*Zhang S, Liu Y, Sun Y, et al.* [[Paper]](https://openreview.net/forum?id=4p6_5HBWPCw) [[GitHub]](https://github.com/snap-research/graphless-neural-networks)
* [**ICLR 2022**] EXACT: Scalable Graph Neural Networks Training via Extreme Activation Compression. 
  >*Liu Z, Zhou K, Yang F, et al.* [[Paper]](https://openreview.net/pdf?id=vkaMaq95_rX)
* [**ICLR 2022**] IGLU: Efficient GCN Training via Lazy Updates. 
  >*Narayanan S D, Sinha A, Jain P, et al.* [[Paper]](https://arxiv.org/pdf/2109.13995)
* [**ICLR 2022**] PipeGCN: Efficient full-graph training of graph convolutional networks with pipelined feature communication. 
  >*Wan C, Li Y, Wolfe C R, et al.* [[Paper]](https://arxiv.org/pdf/2203.10428.pdf) [[GitHub]](https://github.com/RICE-EIC/PipeGCN)
* [**ICLR 2022**] Learn Locally, Correct Globally: A Distributed Algorithm for Training Graph Neural Networks. 
  >*Ramezani M, Cong W, Mahdavi M, et al.* [[Paper]](https://openreview.net/forum?id=FndDxSz3LxQ)
* [**ICML 2022**] Efficient Computation of Higher-Order Subgraph Attribution via Message Passing. 
  >*Xiong et al.* [[Paper]](https://icml.cc/Conferences/2022/Schedule?showEvent=17546)
* [**ICML 2022**] Generalization Guarantee of Training Graph Convolutional Networks with Graph Topology Sampling. 
  >*Li H, Weng M, Liu S, et al.* [[Paper]](https://icml.cc/Conferences/2022/Schedule?showEvent=16764)
* [**ICML 2022**] Scalable Deep Gaussian Markov Random Fields for General Graphs. 
  >*Oskarsson J, Sidén P, Lindsten F.* [[Paper]](https://arxiv.org/abs/2206.05032) [[GitHub]](https://github.com/joeloskarsson/graph-dgmrf)
* [**ICML 2022**] GraphFM: Improving Large-Scale GNN Training via Feature Momentum. 
  >*Yu H, Wang L, Wang B, et al.* [[Paper]](https://arxiv.org/abs/2206.07161) [[GitHub]](https://github.com/divelab/DIG)
* [**SC 2022**] CoGNN: Efficient Scheduling for Concurrent GNN Training on GPUs.
  >*Sun Q, Liu Y, Yang H, et al.* [[Paper]](https://www.computer.org/csdl/proceedings-article/sc/2022/544400a538/1I0bSY3k27S)
* [**MLSys 2022**] BNS-GCN: Efficient Full-Graph Training of Graph Convolutional Networks with Boundary Node Sampling. 
  >*Wan C, Li Y, Li A, et al.* [[Paper]](https://arxiv.org/pdf/2203.10983.pdf) [[GitHub]](https://github.com/RICE-EIC/BNS-GCN)
* [**MLSys 2022**] Graphiler: Optimizing Graph Neural Networks with Message Passing Data Flow Graph. 
  >*Xie Z, Wang M, Ye Z, et al.* [[Paper]](https://proceedings.mlsys.org/paper/2022/hash/a87ff679a2f3e71d9181a67b7542122c-Abstract.html)
* [**MLSys 2022**] Sequential Aggregation and Rematerialization: Distributed Full-batch Training of Graph Neural Networks on Large Graphs. 
  >*Mostafa H.* [[Paper]](https://proceedings.mlsys.org/paper/2022/hash/5fd0b37cd7dbbb00f97ba6ce92bf5add-Abstract.html) [[GitHub]](https://github.com/intellabs/sar)
* [**WWW 2022**] Fograph: Enabling Real-Time Deep Graph Inference with Fog Computing. 
  >*Zeng L, Huang P, Luo K, et al.* [[Paper]](https://dl.acm.org/doi/fullHtml/10.1145/3485447.3511982)
* [**www 2022**] PaSca: A Graph Neural Architecture Search System under the Scalable Paradigm. 
  >*Zhang W, Shen Y, Lin Z, et al.* [[Paper]](https://dl.acm.org/doi/10.1145/3485447.3511986)
* [**www 2022**] Resource-Efficient Training for Large Graph Convolutional Networks with Label-Centric Cumulative Sampling. 
  >*Lin M, Li W, Li D, et al.* [[Paper]](https://dl.acm.org/doi/10.1145/3485447.3512165)
* [**FPGA 2022**] DecGNN: A Framework for Mapping Decoupled GNN Models onto CPU-FPGA Heterogeneous Platform. 
  >*Zhang B, Zeng H, Prasanna V K.* [[Paper]]( https://dl.acm.org/doi/abs/10.1145/3490422.3502326)
* [**FPGA 2022**] HP-GNN: Generating High Throughput GNN Training Implementation on CPU-FPGA Heterogeneous Platform. 
  >*Lin Y C, Zhang B, Prasanna V.* [[Paper]]( https://dl.acm.org/doi/pdf/10.1145/3490422.3502359)
* [**arXiv 2022**] SUGAR: Efficient Subgraph-level Training via Resource-aware Graph Partitioning. 
  >*Xue Z, Yang Y, Yang M, et al.* [[Paper]](https://arxiv.org/pdf/2202.00075.pdf)
* [**CAL 2022**] Characterizing and Understanding Distributed GNN Training on GPUs. 
  >*Lin H, Yan M, Yang X, et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9760056)
* [**ICLR  2021**] Degree-Quant: Quantization-Aware Training for Graph Neural Networks. 
  >*Tailor S A, Fernandez-Marques J, Lane N D.* [[Paper]](https://arxiv.org/abs/2008.05000)
* [**ICLR 2021 Open Review**] FGNAS: FPGA-AWARE GRAPH NEURAL ARCHITECTURE SEARCH. 
  >*Lu Q, Jiang W, Jiang M, et al.* [[Paper]](https://openreview.net/pdf?id=cq4FHzAz9eA)
* [**ICML 2021**] GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training. 
  >*Cai T, Luo S, Xu K, et al.* [[Paper]](http://proceedings.mlr.press/v139/cai21e/cai21e.pdf)
* [**ICML 2021**] Optimization of Graph Neural Networks: Implicit Acceleration by Skip Connections and More Depth. 
  >*Xu K, Zhang M, Jegelka S, et al.* [[Paper]](http://proceedings.mlr.press/v139/xu21k/xu21k.pdf)
* [**KDD 2021**] DeGNN: Improving Graph Neural Networks with Graph Decomposition. 
  >*Miao X, Gürel N M, Zhang W, et al.* [[Paper]](https://dl.acm.org/doi/10.1145/3447548.3467312)
* [**KDD 2021**] Performance-Adaptive Sampling Strategy Towards Fast and Accurate Graph Neural Networks. 
  >*Yoon M, Gervet T, Shi B, et al.* [[Paper]](https://dl.acm.org/doi/10.1145/3447548.3467284)
* [**KDD 2021**] Global Neighbor Sampling for Mixed CPU-GPU Training on Giant Graphs. 
  >*Dong J, Zheng D, Yang L F, et al.* [[Paper]](https://dl.acm.org/doi/10.1145/3447548.3467437)
* [**CVPR  2021**] Binary Graph Neural Networks. 
  >*Bahri M, Bahl G, Zafeiriou S.* [[Paper]](https://arxiv.org/abs/2012.15823)
* [**CVPR  2021**] Bi-GCN: Binary Graph Convolutional Network. 
  >*Wang J, Wang Y, Yang Z, et al.* [[Paper]](https://arxiv.org/abs/2010.07565) [[GitHub]](https://github.com/bywmm/Bi-GCN)
* [**NeurIPS 2021**] Graph Differentiable Architecture Search with Structure Learning. 
  >*Qin Y, Wang X, Zhang Z, et al.* [[Paper]](https://papers.nips.cc/paper/2021/file/8c9f32e03aeb2e3000825c8c875c4edd-Paper.pdf)
* [**ICCAD 2021**] G-CoS: GNN-Accelerator Co-Search Towards Both Better Accuracy and Efficiency. 
  >*Zhang Y, You H, Fu Y, et al.* [[Paper]](https://arxiv.org/pdf/2109.08983.pdf)
* [**GNNSys 2021**] Efficient Data Loader for Fast Sampling-based GNN Training on Large Graphs.
  >*Bai Y, Li C, Lin Z, et al.* [[Paper]](https://gnnsys.github.io/papers/GNNSys21_paper_8.pdf) [[Poster]](https://gnnsys.github.io/posters/GNNSys21_poster_8.pdf)
* [**GNNSys 2021**] Effiicent Distribution for Deep Learning on Large Graphs.
  >*Hoang L, Chen X, Lee H, et al.* [[Paper]](https://gnnsys.github.io/papers/GNNSys21_paper_16.pdf) [[Poster]](https://gnnsys.github.io/posters/GNNSys21_poster_16.pdf)
* [**GNNSys 2021**] Adaptive Filters and Aggregator Fusion for Efficient Graph Convolutions.
  >*Tailor S A, Opolka F L, Lio P, et al.* [[Paper]](https://arxiv.org/abs/2104.01481) [[GitHub]](https://github.com/shyam196/egc)
* [**GNNSys 2021**] Adaptive Load Balancing for Parallel GNN Training.
  >*Su Q, Wang M, Zheng D, et al.* [[Paper]](https://gnnsys.github.io/papers/GNNSys21_paper_18.pdf)
* [**PMLR 2021**] A Unified Lottery Ticket Hypothesis for Graph Neural Networks. 
  >*Chen T, Sui Y, Chen X, et al.* [[Paper]](http://proceedings.mlr.press/v139/chen21p.html)
* [**PVLDB 2021**] Accelerating Large Scale Real-Time GNN Inference using Channel Pruning. 
  >*Zhou H, Srivastava A, Zeng H, et al.* [[Paper]](https://doi.org/10.14778/3461535.3461547) [[GitHub]](https://github.com/tedzhouhk/GCNP)
* [**SC 2021**] Efficient scaling of dynamic graph neural networks.
  >*Chakaravarthy V T, Pandian S S, Raje S, et al.* [[Paper]](https://doi.org/10.1145/3458817.3480858)
* [**RTAS 2021**] Optimizing Memory Efficiency of Graph Neural Networks on Edge Computing Platforms. 
  >*Zhou A, Yang J, Gao Y, et al.* [[Paper]](https://arxiv.org/abs/2104.03058) [[GitHub]](https://github.com/BUAA-CI-Lab/GNN-Feature-Decomposition)
* [**ICDM 2021**] GraphANGEL: Adaptive aNd Structure-Aware Sampling on Graph NEuraL Networks. 
  >*Peng J, Shen Y, Chen L.* [[Paper]](https://ieeexplore.ieee.org/document/9679081)
* [**GLSVLSI 2021**] Co-Exploration of Graph Neural Network and Network-on-Chip Design Using AutoML. 
  >*Manu D, Huang S, Ding C, et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3453688.3461741)
* [**arXiv 2021**] Edge-featured Graph Neural Architecture Search.
  >*Cai S, Li L, Han X, et al.* [[Paper]](http://arxiv.org/abs/2109.01356)
* [**arXiv 2021**] GNNSampler: Bridging the Gap between Sampling Algorithms of GNN and Hardware. 
  >*Liu X, Yan M, Song S, et al.* [[Paper]](https://arxiv.org/abs/2108.11571v1) [[GitHub]](https://github.com/temp-gimlab/gnnsampler)
* [**KDD  2020**] TinyGNN: Learning Efficient Graph Neural Networks. 
  >*Yan B, Wang C, Guo G, et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3394486.3403236)
* [**ICLR 2020**] GraphSAINT: Graph Sampling Based Inductive Learning Method. 
  >*Zeng H, Zhou H, Srivastava A, et al.* [[Paper]](https://arxiv.org/pdf/1907.04931) [[GitHub]](https://github.com/GraphSAINT/GraphSAINT)
* [**NeurIPS 2020**] Gcn meets gpu: Decoupling “when to sample” from “how to sample”.
  >*Ramezani M, Cong W, Mahdavi M, et al.* [[Paper]](https://proceedings.neurips.cc/paper/2020/file/d714d2c5a796d5814c565d78dd16188d-Paper.pdf)
* [**SC 2020**] Reducing Communication in Graph Neural Network Training. 
  >*Tripathy A, Yelick K, Buluç A.* [[Paper]](https://arxiv.org/abs/2005.03300) [[GitHub]](https://github.com/PASSIONLab/gnn_training)
* [**ICTAI 2020**] SGQuant: Squeezing the Last Bit on Graph Neural Networks with Specialized Quantization. 
  >*Feng B, Wang Y, Li X, et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9288186)
* [**arXiv 2020**] Learned Low Precision Graph Neural Networks. 
  >*Zhao Y, Wang D, Bates D, et al.* [[Paper]](https://www.euromlsys.eu/pub/zhao21euromlsys.pdf)
* [**arXiv 2020**] Distributed Training of Graph Convolutional Networks using Subgraph Approximation.
  >*Angerd A, Balasubramanian K, Annavaram M.* [[Paper]](https://arxiv.org/abs/2012.04930)
* [**IPDPS 2019**] Accurate, efficient and scalable graph embedding. 
  >*Zeng H, Zhou H, Srivastava A, et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/8820993)
---
## Surveys and Performance Analysis on Graph Learning
* [**TKDE 2025**] Acceleration Algorithms in GNNs: A Survey.
 >*Ma L, Sheng Z, Li X, et al.* [[paper]]([https://fangmingliu.github.io/files/WWW24-LambdaGrapher.pdf](https://arxiv.org/pdf/2405.04114))
* [**TPDS 2025**] Survey on Characterizing and Understanding GNNs from a Computer Architecture Perspective.
  >*Meng Wu, Mingyu Yan, Wenming Li, et al.* [[Paper]](https://doi.org/10.1109/TPDS.2025.3532089)
* [**CAL 2023**] Architectural Implications of GNN Aggregation Programming Abstractions.  
  >*Qi Y, Yang J, Zhou A, et al.* [[Paper]](https://ieeexplore.ieee.org/document/10288038)
* [**arXiv 2023**] A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware.  
  >*S Zhang, A Sohrabizadeh, C Wan, et al.* [[Paper]](https://arxiv.org/pdf/2306.14052)
* [**arXiv 2022**] A Comprehensive Survey on Distributed Training of Graph Neural Networks.  
  >*Lin H, Yan M, Ye X, et al.* [[Paper]](https://arxiv.org/abs/2211.05368)
* [**arXiv 2022**] Distributed Graph Neural Network Training: A Survey.
  >*Shao Y, Li H, Gu X, et al.* [[Paper]](https://arxiv.org/abs/2211.00216)
* [**CAL 2022**] Characterizing and Understanding HGNNs on GPUs. 
  >*Yan M, Zou M, Yang X, et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9855397/)
* [**arXiv 2022**] Parallel and Distributed Graph Neural Networks: An In-Depth Concurrency Analysis. 
  >*Besta M, Hoefler T.* [[Paper]](https://arxiv.org/abs/2205.09702)
* [**IJCAI 2022**] Survey on Graph Neural Network Acceleration: An Algorithmic Perspective. 
  >*Liu X, Yan M, Deng L, et al.* [[Paper]](https://arxiv.org/pdf/2202.04822)
* [**ACM Computing Surveys 2022**] A Practical Tutorial on Graph Neural Networks
  >*Ward I R, Joyner J, Lickfold C, et al.*[[Paper]](https://dl.acm.org/doi/10.1145/3503043)
* [**CAL 2022**] Characterizing and Understanding Distributed GNN Training on GPUs.
  >*Lin H, Yan M, Yang X, et al.*[[Paper]](https://dl.acm.org/doi/10.1145/3503043)
* [**Access 2022**] Analyzing GCN Aggregation on GPU. 
  >*Kim I, Jeong J, Oh Y, et al.*[[Paper]](https://ieeexplore.ieee.org/iel7/6287639/9668973/09930519.pdf)
* [**GNNSys 2021**] Analyzing the Performance of Graph Neural Networks with Pipe Parallelism.
  >*Dearing M T, Wang X.* [[Paper]](https://gnnsys.github.io/papers/GNNSys21_paper_12.pdf) [[Poster]](https://gnnsys.github.io/posters/GNNSys21_poster_12.pdf)
* [**IJCAI 2021**] Automated Machine Learning on Graphs: A Survey. 
  >*Zhang Z, Wang X, Zhu W.* [[Paper]](https://arxiv.org/abs/2103.00742)
* [**PPoPP 2021**] Understanding and bridging the gaps in current GNN performance optimizations. 
  >*Huang K, Zhai J, Zheng Z, et al.* [[Paper]](https://dl.acm.org/doi/abs/10.1145/3437801.3441585)
* [**ISCAS 2021**] Characterizing the Communication Requirements of GNN Accelerators: A Model-Based Approach. 
  >*Guirado R, Jain A, Abadal S, et al.* [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9401612)
* [**ISPASS  2021**] GNNMark: A Benchmark Suite to Characterize Graph Neural Network Training on GPUs. 
  >*Baruah T, Shivdikar K, Dong S, et al.* [[Paper]](https://ieeexplore.ieee.org/abstract/document/9408205)
* [**ISPASS  2021**] Performance Analysis of Graph Neural Network Frameworks. 
  >*Wu J, Sun J, Sun H, et al.* [[Paper]](https://www.semanticscholar.org/paper/Performance-Analysis-of-Graph-Neural-Network-Wu-Sun/b6da3ab0a6e710f16e11e5890818a107d1d5735c)
* [**CAL 2021**] Making a Better Use of Caches for GCN Accelerators with Feature Slicing and Automatic Tile Morphing. 
  >*Yoo M, Song J, Lee J, et al.* [[Paper]](https://ieeexplore.ieee.org/document/9461595/)
* [**arXiv 2021**] Understanding GNN Computational Graph: A Coordinated Computation, IO, and Memory Perspective. 
  >*Zhang H, Yu Z, Dai G, et al.* [[Paper]](http://arxiv.org/abs/2110.09524)
* [**arXiv 2021**] Understanding the Design Space of Sparse/Dense Multiphase Dataflows for Mapping Graph Neural Networks on Spatial Accelerators. 
  >*Garg R, Qin E, Muñoz-Martínez F, et al.* [[Paper]](https://arxiv.org/pdf/2103.07977)
* [**arXiv 2021**] A Taxonomy for Classification and Comparison of Dataflows for GNN Accelerators. 
  >*Garg R, Qin E, Martínez F M, et al.* [[Paper]](https://arxiv.org/abs/2103.07977)
* [**arXiv  2021**] Graph Neural Networks: Methods, Applications, and Opportunities. 
  >*Waikhom L, Patgiri R.* [[Paper]](http://arxiv.org/abs/2108.10733)
* [**arXiv 2021**] Sampling methods for efficient training of graph convolutional networks: A survey. 
  >*Liu X, Yan M, Deng L, et al.* [[Paper]](https://arxiv.org/abs/2103.05872)
* [**KDD 2020**] Deep Graph Learning: Foundations, Advances and Applications. 
  >*Rong Y, Xu T, Huang J, et al.* [[Paper]](https://dl.acm.org/doi/10.1145/3394486.3406474)
* [**TKDE 2020**] Deep Learning on Graphs: A Survey. 
  >*Zhang Z, Cui P, Zhu W.*[[paper]](https://ieeexplore.ieee.org/abstract/document/9039675)
* [**CAL 2020**] Characterizing and Understanding GCNs on GPU. 
  >*Yan M, Chen Z, Deng L, et al.* [[Paper]](https://arxiv.org/abs/2010.00130)
* [**arXiv 2020**] Computing Graph Neural Networks: A Survey from Algorithms to Accelerators. 
  >*Abadal S, Jain A, Guirado R, et al.* [[Paper]](https://arxiv.org/abs/2010.00130)











---
## Maintainers
- Ao Zhou, Beihang University. [[GitHub]](https://github.com/ZhouAo-ZA)
- Yingjie Qi, Beihang University. [[GitHub]](https://github.com/Kevin7Qi)
- Tong Qiao, Beihang University. [[GitHub]](https://github.com/qiaotonggg)



