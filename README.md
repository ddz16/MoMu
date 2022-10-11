# MoMu

The PyTorch implementation of MoMu, described in "Natural Language-informed Understanding of Molecule Graphs".

Our collected dataset consists of two folders holding molecular graphs and texts, respectively. The dataset can be downloaded in [https://pan.baidu.com/s/1aHJoYTTZWDHPCcRuu9I7Fg](https://pan.baidu.com/s/1aHJoYTTZWDHPCcRuu9I7Fg), the passward is **1234**. We will introduce our pretrain method and all the downstream tasks below. Please refer to the ReadMe in each part for the source code, system requirements, installation, demo, instructions for use, etc.

# Pretrain

The pretrain code is available in the `Pretrain/` folder.

Our pretrained models MoMu-S and MoMu-K can be downloaded on [the Baidu Netdisk](https://pan.baidu.com/s/1jvMP_ysQGTMd_2sTLUD45A), the password is **1234**. All the downstream tasks use these two models.

```python
MoMu-K:   checkpoints/littlegin=graphclinit_bert=kvplm_epoch=299-step=18300.ckpt
MoMu-S:   checkpoints/littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt
```

# Cross-modality retrieval

Since our MoMu model is pre-trained by matching weakly-correlated texts to corresponding molecular graphs, it is able to process both the graph and text modalities of molecules. We evaluate its performance in cross-modality retrieval. Given a molecule graph, graph-to-text (G-T) retrieval aims to retrieve the most relevant text descriptions of this molecule. Conversely, given a text paragraph, text-to-graph (T-G) retrieval aims at retrieving the most relevant molecule graph it describes. The code for these two downstream tasks is available in the `GraphTextRetrieval/` folder.

# Molecule caption

The molecule captioning task aims to generate texts to describe the given molecule. The code for this downstream task is available in the `MoleculeCaption/` folder.

# Zero-shot text-to-graph molecule generation

We propose a new task called zero-shot text-to-graph molecule generation. The goal is to design a cross modality molecule generator that takes as input the natural language description of the desired conditions and imagines new molecules that match the description. The code for this downstream task is available in the `Text2graph/` folder.

# Molecule property prediction

Molecular property prediction is a graph-level prediction task that is usually used to evaluate the transfer ability of pre-trained graph encoders. The code for this downstream task is available in the `MoleculePrediction/` folder.

# Citation

```
@article{su2022molecular,
  title={Natural Language-informed Understanding of Molecule Graphs},
  author={Bing Su, Dazhao Du, Zhao Yang, Yujie Zhou, Jiangmeng Li, Anyi Rao, Hao Sun, Zhiwu Lu, Ji-Rong Wen},
  year={2022}
}
```