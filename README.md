# CNN-RNN2016
Reimplementation the paper of Recurrent Convolutional Network for Video-based Person Re-Identification in Pytorch
# Preparation
Python 3.6  
Pytorch >= 0.4.0
# Result on Prid2011

|版本| map| rank1  | rank5 | rank10  | rank20  |
| :---: | :---: |:-------:|:---:  |:------:| -------:|
| 复现 | 58.8%| 49.4% | 68.5% | 83.1% | 89.9%  |
| 原文 | --| 70% | 90% | 95% | 97%  |


# Problems
P.1 use the official split to form dataset, the dataset is too small.

  train identites: 89, test identites: 89

  => PRID-2011 loaded
  
 |subset |# ids| # tracklets  |
 | :---: | :---: |:-------:|
 | train  | 89| 178 | 
 | query  | 89| 89 | 
 | gallery  | 89| 89 |
 | total   |  178| 356 |

P.2 Data Augmentation

    1.mirror is not implemented
    2.resize the image size from (128, 64) to (256, 128),not same as (64,48) in offical code

P.3 Training
 
    1.If batch-size set to 1, the net will not be convergent.
    2.The dataset is too small, we can change the dataset generation way to extend 
      the dataset. Maybe like the paper 'Video Person Re-identification with
      Competitive Snippet-similarity Aggregation and Co-attentive Snippet Embedding'.
  
  # Reference
[Recurrent-Convolutional-Video-ReID](https://github.com/niallmcl/Recurrent-Convolutional-Video-ReID)

[Spatial-Temporal-Pooling-Networks-ReID](https://github.com/YuanLeung/Spatial-Temporal-Pooling-Networks-ReID)
