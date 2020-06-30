# Fontfusion
Simple Tensorflow implementation of Fontfusion. Please refer to our following paper for algorithm details: Learn to Design a New Font by Fusing Different Calligraphy.

1. Requirements
Tensorflow 1.8
Python 3.6

2. Usage:
├── dataset
   └── YOUR_DATASET_NAME
       ├── trainA
           ├── xxx.jpg (name, format doesn't matter)
           ├── yyy.png
           └── ...
       ├── trainB
           ├── zzz.jpg
           ├── www.png
           └── ...
       ├── testA
           ├── aaa.jpg 
           ├── bbb.png
           └── ...
       └── testB
           ├── ccc.jpg 
           ├── ddd.png
           └── ...
           
3. Train
python main.py --phase train --dataset pose --epoch 5

4. Test
python main.py --phase test --dataset pose --epoch 1
