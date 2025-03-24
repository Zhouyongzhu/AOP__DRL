# AOP-DRL

## About

AOP-DRL: An antioxidant peptide recognition method based on deep representation learning

## Datasets

The dataset used in this study is publicly available in the AnOxPePred repository: \\
\url{<https://github.com/TobiasHeOl/AnOxPePred}>.

## Hardware Requirements

GPU: 2× NVIDIA Tesla V100 (32GB VRAM)
Cluster: National Protein Science Center GPU Cluster Node

### Software Dependencies

```bash
conda create -n aopdrl python=3.9
conda install -c pytorch pytorch=2.2 torchvision cudatoolkit=11.8
pip install -r requirements.txt
```

## License

AOP-DRL is for non-commercial use only.
