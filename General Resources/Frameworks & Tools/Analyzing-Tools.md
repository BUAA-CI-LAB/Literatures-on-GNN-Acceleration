# Profiling and Analyzing Tools

- [Pytorch Profile Tool](#Pytorch-Profile-Tool-Documentation-Tutorial)
- [NVIDIA Nsight Compute](#NVIDIA-Nsight-Compute-Documentation-Tutorial)
- [Linux Perf](#Linux-Perf-Documentation-Tutorial)

## Pytorch Profile Tool [[Documentation]](https://pytorch.org/docs/stable/profiler.html) [[Tutorial]](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

PyTorch Profiler is a tool that allows the collecton of the performance metrics during the training and inference. Profiler’s context manager API can be used to better understand what model operators are the most expensive, examine their input shapes and stack traces, study device kernel activity and visualize the execution trace.

## NVIDIA Nsight Compute [[Documentation]](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html) [[Tutorial]](https://developer.nvidia.com/nsight-compute-blogs)

NVIDIA Nsight Compute is an interactive kernel profiler for CUDA applications. It provides detailed performance metrics and API debugging via a user interface and command line tool. In addition, its baseline feature allows users to compare results within the tool. Nsight Compute provides a customizable and data-driven user interface and metric collection and can be extended with analysis scripts for post-processing results.

## Linux Perf [[Documentation]](https://en.wikipedia.org/wiki/Perf_(Linux)) [[Tutorial]](https://www.brendangregg.com/perf.html)

perf (sometimes called perf_events or perf tools, originally Performance Counters for Linux, PCL) is a performance analyzing tool in Linux, available from Linux kernel version 2.6.31 in 2009. Userspace controlling utility, named perf, is accessed from the command line and provides a number of subcommands; it is capable of statistical profiling of the entire system (both kernel and userland code). It supports hardware performance counters, tracepoints, software performance counters (e.g. hrtimer), and dynamic probes (for example, kprobes or uprobes). In 2012, two IBM engineers recognized perf (along with OProfile) as one of the two most commonly used performance counter profiling tools on Linux.

