# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 19:21:59 2020

@author: QMX
"""

#encoding: utf-8
import os
import pygame

pygame.init()


crop_size = 256  

#Training set
chinese_dir = 'dataset/pose/trainA'
if not os.path.exists(chinese_dir):
    os.mkdir(chinese_dir)

with open("font/train.txt", "r") as f:
    data = f.read()
    for i in range(4700):
        font = pygame.font.Font("font/Slender_Gold.ttf", crop_size)
        rtext = font.render(data[i], True, (0, 0, 0), (255, 255, 255))
        newimg = pygame.transform.scale(rtext, (crop_size, crop_size)) 
        pygame.image.save(newimg, os.path.join(chinese_dir, str(i) + ".png"))

chinese_dir = 'dataset/pose/trainB'
if not os.path.exists(chinese_dir):
    os.mkdir(chinese_dir)

with open("font/train.txt", "r") as f:
    data = f.read()
    for i in range(4700):
        font = pygame.font.Font("font/Running_Hand.TTF", crop_size)
        rtext = font.render(data[i], True, (0, 0, 0), (255, 255, 255))
        newimg = pygame.transform.scale(rtext, (crop_size, crop_size)) 
        pygame.image.save(newimg, os.path.join(chinese_dir, str(i) + ".png"))



#Testing set  
chinese_dir = 'dataset/pose/testA'
if not os.path.exists(chinese_dir):
    os.mkdir(chinese_dir)

with open("font/test.txt", "r") as f:
    data = f.read()
    for i in range(765):
        font = pygame.font.Font("font/Slender_Gold.ttf", crop_size)
        rtext = font.render(data[i], True, (0, 0, 0), (255, 255, 255))
        newimg = pygame.transform.scale(rtext, (crop_size, crop_size)) 
        pygame.image.save(newimg, os.path.join(chinese_dir, str(i) + ".png"))


chinese_dir = 'dataset/pose/testB'
if not os.path.exists(chinese_dir):
    os.mkdir(chinese_dir)

with open("font/test.txt", "r") as f:
    data = f.read()
    for i in range(765):
        font = pygame.font.Font("font/Running_Hand.TTF", crop_size)
        rtext = font.render(data[i], True, (0, 0, 0), (255, 255, 255))
        newimg = pygame.transform.scale(rtext, (crop_size, crop_size)) 
        pygame.image.save(newimg, os.path.join(chinese_dir, str(i) + ".png"))