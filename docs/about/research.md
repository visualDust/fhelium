---
title: Research
description: Software citation, research publications, and cross-layer methods that shaped FHElium.
---

# Research

FHElium grew out of research on cross-layer optimization for encrypted machine learning. Some of this work was developed during early prototype stages, before the current project name and architecture were established, but its central ideas continue to shape FHElium.

The shared theme is that encrypted applications should not be optimized one layer at a time. Application structure can guide encrypted representation, parallel strategy, communication, memory use, and hardware execution; observations from lower layers can then reshape decisions above them.

## Collaborate with us

We welcome research collaborations on FHE algorithms, compilers, numerical methods, runtime systems, distributed execution, benchmarking, and hardware acceleration. FHElium is intended to support work that studies one layer in depth or follows its effects across the encrypted-execution stack. Researchers and organizations interested in developing, evaluating, or integrating related ideas are invited to [contact us](mailto:zgong6@ncsu.edu).

If you use FHElium as software, please cite the repository, see also [`CITATION.cff`](https://github.com/VisualDust/fhelium/blob/main/CITATION.cff) or cite the following BibTeX entry:

```bibtex
@misc{gong2026fhelium,
  title        = {FHElium: A Cross-Stack CKKS Research Framework for CPU and CUDA},
  author       = {Gong, Zhaoting and Liang, Jiaming and Ran, Ran and Wen, Wujie},
  year         = {2026},
  month        = jul,
  note         = {Version 0.10.0},
  url          = {https://github.com/VisualDust/fhelium}
}
```

## AEGIS

### Scaling Long-Sequence Homomorphic Encrypted Transformer Inference via Hybrid Parallelism on Multi-GPU Systems

**Zhaoting Gong, Ran Ran, Fan Yao, and Wujie Wen**

*Proceedings of the 40th ACM International Conference on Supercomputing (ICS 2026), pages 1206–1219*

[DOI: 10.1145/3797905.3800539](https://doi.org/10.1145/3797905.3800539)

AEGIS studies long-sequence encrypted Transformer inference across multiple GPUs. It identifies cross-GPU data dependencies from the application, derives a hybrid parallel strategy from those dependencies, and inserts encryption-aware communication into the resulting execution plan.

This is a cross-layer optimization: application semantics determine how encrypted work should be partitioned, while communication and hardware constraints feed back into the strategy selected above them. These ideas became part of FHElium's program-transformation design: application-level dependency analysis can guide parallel planning and materialize encryption-aware communication within the program rather than leaving distribution as a disconnected runtime concern.

```bibtex
@inproceedings{gong2026scaling,
  title={Scaling Long-Sequence Homomorphic Encrypted Transformer Inference via Hybrid Parallelism on Multi-GPU Systems},
  author={Gong, Zhaoting and Ran, Ran and Yao, Fan and Wen, Wujie},
  booktitle={Proceedings of the 40th ACM International Conference on Supercomputing},
  pages={1206--1219},
  year={2026},
  doi={10.1145/3797905.3800539}
}
```

## G-HEMP

### Fast Multi-GPU Private Inference for Large-Scale GCNs with Homomorphic Encryption

**Ran Ran, Zhaoting Gong, Zhaowei Li, Xianting Lu, Jiajia Li, and Wujie Wen**

*The Ninth Conference on Machine Learning and Systems (MLSys 2026), Oral*

[Official presentation and abstract](https://mlsys.org/virtual/2026/oral/3811)

G-HEMP studies large-scale private inference for graph convolutional networks. It combines a block-diagonal parallel packing method with application-aware multi-GPU workload partitioning, reducing redundant encrypted work and distributing both computation and memory pressure across devices.

G-HEMP uses FHElium's multi-GPU execution layer to realize its application-specific schedule. Its packing strategy, graph structure, workload partition, memory behavior, and device execution are designed together, turning multi-GPU support into an algorithmic optimization opportunity rather than a transparent replication mechanism.

```bibtex
@inproceedings{ran2026g,
  title={{G-HEMP}: Fast Multi-GPU Private Inference for Large-Scale {GCNs} with Homomorphic Encryption},
  author={Ran, Ran and Gong, Zhaoting and Li, Zhaowei and Lu, Xianting and Li, Jiajia and Wen, Wujie},
  booktitle={Ninth Conference on Machine Learning and Systems},
  year={2026}
}
```
