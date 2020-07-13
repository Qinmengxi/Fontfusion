# FontfusionNet
Simple Tensorflow implementation of Fontfusion. More details can be referred in our paper: Learn to Design a New Font by Fusing Different Calligraphy, which is under review.

## 1. Requirements
Tensorflow 1.8 <br>
Python 3.6 <br>
pygame <br>

## 2. Usage:
├── dataset <br>
&emsp;&emsp;└── YOUR_DATASET_NAME <br>
&emsp;&emsp;&emsp;&emsp;├── trainA <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── xxx.jpg (name, format doesn't matter) <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── yyy.png <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;└── ... <br>
&emsp;&emsp;&emsp;&emsp;├── trainB <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── zzz.jpg <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── www.png <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;└── ... <br>
&emsp;&emsp;&emsp;&emsp;├── testA <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── aaa.jpg <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── bbb.png <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;└── ... <br>
&emsp;&emsp;&emsp;&emsp;└── testB <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── ccc.jpg <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;├── ddd.png <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;└── ... <br>
           
## 3. Train
python main.py --phase train --dataset pose --epoch 5 <br>

## 4. Test
python main.py --phase test --dataset pose --epoch 1 <br>
