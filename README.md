# CHIRON
This is the official repository for the EMNLP 2024 Findings paper ["CHIRON: Rich Character Representations in Long-Form Narratives"](https://arxiv.org/abs/2406.10190) by Alex Gurung and Mirella Lapata. In the paper we propose a new character-sheet based representation, CHIRON, useful for long-story tasks that require character understanding and for story analysis. The code in this repository will allow you to create your own character sheets, and calculate the _density_ character-centricity metric we propose. Refer to the paper for more details.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Citation](#citation)

## Installation

(Code was tested using docker image [pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel](https://hub.docker.com/layers/pytorch/pytorch/2.5.0-cuda12.4-cudnn9-devel/images/sha256-704c91d25b97109faf63f3d5ef33744c3684032ab3d10536fab4712bb764f8f6?context=explore))
```shell
$ conda create -n chiron python=3.8
$ conda activate chiron
$ pip install -r requirements.txt
$ python end_to_end_chiron_from_snippet.py
```

## Usage

For creating your own character sheets from a snippet, run

`python end_to_end_chiron_from_snippet.py`


## Dataset

We build the paper's work on three data sources: [STORIUM](https://aclanthology.org/2020.emnlp-main.525/), [New Yorker/Art or Artifice?](https://dl.acm.org/doi/10.1145/3613904.3642731), and [DOC/RE<sup>3</sup>](https://aclanthology.org/2023.acl-long.190/). Due to copyright concerns we cannot release this data here, but the instructions for reconstructing the datasets are present in the paper. As of writing the full STORIUM dataset is available [here](https://storium.cs.umass.edu/). For academic research purposes only you may also reach out to Alex Gurung for a subset of the data that we used for this paper.

## Citation
```bibtex
@article{gurung2024chiron,
  title={CHIRON: Rich Character Representations in Long-Form Narratives},
  author={Gurung, Alexander and Lapata, Mirella},
  journal={arXiv preprint arXiv: 2406.10190},
  year={2024}
}
```
