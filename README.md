# HACAN
The codes for our paper "HACAN: Hybrid Attention-Driven Cross-Layer Alignment Network for Image-Text Retrieval". It is built on top of the [HAT](https://github.com/LuminosityX/HAT?tab=readme-ov-file).

## Introduction
In the field of image-text matching and cross-modal retrieval, although some advancements have been made in fine-grained retrieval techniques, current methods tend to concentrate only on the direct connections between visual elements of images and textual keywords. This approach neglects the more complicated semantic interactions between modalities, both at the local and global levels, resulting in semantic ambiguity. 
![HACAN](overview.png)
We introduce HACAN, a Hybrid Attention-Driven Cross-layer Alignment Network, that uses BERT and ConvNeXt to integrate global and local strategies, addressing issues of semantic ambiguity and misalignment. With the proposed global contrastive divergence loss, HACAN enhances the complementarity between vision and language, improving the model's ability to distinguish between positive and negative samples. HACAN significantly enhances retrieval efficiency by incorporating hierarchical inference strategies. On the Flickr30K and MS-COCO datasets, HACAN outperforms state-of-the-art image-text retrieval methods by 5% to 8% in the Rsum metric.

## Preparation
### Dependencies
We recommended to use Anaconda for the following packages.
- python >= 3.8
- [torch](http://pytorch.org/) (>=1.8.1)
- [lightning](https://lightning.ai/)(1.8.0)
- [transformers](https://huggingface.co/docs/transformers) (4.24.0)
- torchvision
- opencv-python

### Data
The experimental dataset can be downloaded from [Flickr30K](http://shannon.cs.illinois.edu/DenotationGraph/) and [MSCOCO](http://mscoco.org/). We will subsequently release the experimental pre-trained model for public access. We refer to the path of extracted files as `$DATASET_PATH`and the storage location of the pre-trained model as `$MODEL_PATH`.

## Evaluation
Run `run.py` to evaluate the trained models on Flickr30K or MSCOCO.
```bash
Test on Flickr30K:
python run.py with data_root=`$DATASET_PATH` test_only=True checkpoint=`$MODEL_PATH`

Test on MSCOCO:
python run.py with coco_config data_root=`$DATASET_PATH` test_only=True checkpoint=`$MODEL_PATH`
```

## Training
Run `run.py` to train the model on Flickr30K or MSCOCO.
```bash
Train on Flickr30K:
python run.py with data_root=`$DATASET_PATH` loss="GCD" 

Train on MSCOCO:
python run.py with coco_config data_root=`$DATASET_PATH` loss="GCD"
```

