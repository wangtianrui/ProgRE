# This folder primarily contains auxiliary materials for our paper

* Analysis of Migration Errors: Analyzes errors encountered when migrating a model (demonstrated with a HuBERT Base model) from the MindSpore framework to the PyTorch framework, using calculations and visualizations.
* Chinese Speech Recognition Performance: Demonstrates the performance of the proposed model on the Chinese speech recognition task.


## Analysis of Migration Errors
We directly convert the parameters of the HuBERT model trained based on the MindSpore framework into a format readable by PyTorch. Then, using the same parameters, we extract features from the same audio using HuBERT built with MindSpore and HuBERT built with PyTorch, respectively. We saved the input to the encoder and the output of each layer of the encoder, totaling 13 layers of representations for comparison.

First, we visualize the representations of the 13 layers and the error. The first column is the result from MindSpore, the second column is the result from PyTorch, and the third column is the error with the formula `diff = abs(ms - pt)`.

![](./supplementary_results/test.png)

Next, we show the relative error, calculated as `np.mean(abs(ms - pt)) / np.mean(abs(ms))`, with the result being as follow. 
```txt
0 0.0005469106
1 0.0008232473
2 0.0015528084
3 0.0020986304
4 0.00258094
5 0.0024200461
6 0.0024519553
7 0.0021557575
8 0.0020926956
9 0.0019156568
10 0.002052982
11 0.0021666065
12 0.002720383
```
It can be seen that the migration introduces approximately a 0.25% relative error.
This causes a slight performance degradation when the model pre-trained based on MindSpore is used in a fine-tuning framework based on PyTorch.



## Chinese Speech Recognition Performance