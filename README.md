# Fontfusion
Simple Tensorflow implementation of Fontfusion. More details can be referred in our paper which is under review.

## 1.Requirements
* Tensorflow 1.8 <br>
* Python 3.6 <br>
* pygame <br>
* CUDA <br>
* cudnn <br>

## 2.Create datasets:
You can create different datasets for font fusion by replacing font in datasets.py.
* python datasets.py <br>

## 3.Proposed network:
![generator](https://github.com/Qinmengxi/Fontfusion/blob/master/figure/netwowk.png)
![discriminator](https://github.com/Qinmengxi/Fontfusion/blob/master/figure/discriminator.png)

## 4.Usage:
├── dataset <br>
&emsp;&emsp;└── YOUR_DATASET_NAME <br>
&emsp;&emsp;&emsp;&emsp;├── trainA <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── 1.jpg (name, format doesn't matter) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── 2.png <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;└── ... <br>
&emsp;&emsp;&emsp;&emsp;├── trainB <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── 1.jpg <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── 2.png <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;└── ... <br>
&emsp;&emsp;&emsp;&emsp;├── testA <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── 1.jpg <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── 2.png <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;└── ... <br>
&emsp;&emsp;&emsp;&emsp;└── testB <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── 1.jpg <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── 2.png <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;└── ... <br>
           
## 5.Train

* python main.py --phase train --dataset pose --epoch 5 <br>

## 6.Test
* python main.py --phase test --dataset pose --epoch 1 <br>

## 7.Result
![result](https://github.com/Qinmengxi/Fontfusion/blob/master/figure/result.png)

## 8.Reference
* Joo D, Kim D, Kim J. Generating a fusion image: One’s identity and another’s shape. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, Salt Lake City, USA, pp.1635-1643, 2018. DOI:10.1109/CVPR.2018.00176. <br>
* Huang X, Liu M Y, Belongie S, et al. Multimodal unsupervised image-to-image translation. In Proceedings of the European Conference on Computer Vision, Munich, Germany, pp.179-196, 2018. DOI: 10.1007/978-3-030-01219-9 11. <br>
