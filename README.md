# HACAN
The codes for our paper "HACAN: Hybrid Attention-Driven Cross-Layer Alignment Network for Image-Text Retrieval". It is built on top of the [HAT](https://github.com/LuminosityX/HAT?tab=readme-ov-file).

## Introduction
xxx

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
The experimental dataset can be downloaded from [Flickr30K](http://shannon.cs.illinois.edu/DenotationGraph/) and [MSCOCO](http://mscoco.org/). The pre-trained model can be downloaded from here. We refer to the path of extracted files as `$DATASET_PATH`and the storage location of the pre-trained model as `$MODEL_PATH`.

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

