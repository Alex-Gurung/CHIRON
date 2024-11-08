# CHIRON
This is the official repository for the EMNLP 2024 Findings paper ["CHIRON: Rich Character Representations in Long-Form Narratives"](https://arxiv.org/abs/2406.10190) by Alex Gurung and Mirella Lapata. In the paper we propose a new character-sheet based representation, CHIRON, useful for long-story tasks that require character understanding and for story analysis. The code in this repository will allow you to create your own character sheets, and calculate the _density_ character-centricity metric we propose. Refer to the paper for more details. 

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Citation](#citation)

## Installation

(Code was tested using docker image [pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel](https://hub.docker.com/layers/pytorch/pytorch/2.5.1-cuda12.1-cudnn9-devel/images/sha256-e8e63dd7baca894ba11fe1ba48a52a550793c8974f89b533d697784dd20a4dc0?context=explore))
```shell
$ conda create -n chiron python=3.11 # might be unnecessary if you are using a docker image
$ conda activate chiron
$ pip install -r requirements.txt
$ huggingface-cli login --token HUGGINFACE_TOKEN # get your token from https://huggingface.co/settings/tokens
$ python -m spacy download en_core_web_lg # can use different model if you want
$ pip install vllm==0.6.3.post1
```

Note: vllm sometimes has breaking changes, we used `0.5.4` for the paper (with beam search) but as it is no longer supported this code uses the latest available version as of writing.

## Usage


### From Snippet
For creating your own character sheets from a snippet, run

`python end_to_end_chiron_from_snippet.py`

We provide a sample snippet and default parameters, run `python end_to_end_chiron_from_snippet.py --help` for more details.

### From Dataset

For creating character sheets from a dataset, run

`python end_to_end_chiron_from_dataset.py` (see `python end_to_end_chiron_from_dataset.py --help` for more details on parameters).

This will output a character sheet to the provided directory for each story-character pair in the dataset. There will be three files for each story-character pair:

- `story_id-character_unfiltered.txt`: the character sheet without the verification classifier filtering
- `story_id-character_postfiltering.txt`: the character sheet after verification classifier filtering
- `story_id-character_postfilteringdedup.txt`: the character sheet after verification classifier filtering and statement deduplication

This last file is the one you should use for further analysis/downstream tasks.

Your dataset of stories and snippets should be in the following format:
```
{
    "story_id": str, # unique identifier for the story
    "character": str, # character sheet name
    "snippets": List[List[str]]
}
```

`snippets` should be a list of snippets from the story, where each snippet is either a list of `[snippet_role, original_text, snippet]` or just `[original_text, snippet]` - this depends on dataset and allows you to split snippets to a normal size while maintaining references to the original snippet snippet role.


## Dataset

We build the paper's work on three data sources: [STORIUM](https://aclanthology.org/2020.emnlp-main.525/), [New Yorker/Art or Artifice?](https://dl.acm.org/doi/10.1145/3613904.3642731), and [DOC/RE<sup>3</sup>](https://aclanthology.org/2023.acl-long.190/). Due to copyright concerns we cannot release this data here, but the instructions for reconstructing the datasets are present in the paper. As of writing the full STORIUM dataset is available [here](https://storium.cs.umass.edu/). For academic research purposes only you may also reach out to Alex Gurung for a subset of the STORIUM data that we used for this paper.

## Citation
```bibtex
@article{gurung2024chiron,
  title={CHIRON: Rich Character Representations in Long-Form Narratives},
  author={Gurung, Alexander and Lapata, Mirella},
  journal={arXiv preprint arXiv: 2406.10190},
  year={2024}
}
```
