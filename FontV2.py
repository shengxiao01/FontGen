# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
'''
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import numpy as np

letter = 'A'
size = 50

font = ImageFont.truetype('B691-Sans-Heavy.ttf', size*2)

current_canvas_size = size * 3



while True:
    img = Image.new('L', (current_canvas_size,current_canvas_size),(0))
    draw = ImageDraw.Draw(img)
    draw.text((current_canvas_size/6,current_canvas_size/6),letter,(1),font = font)
    draw = ImageDraw.Draw(img)
    left,upper,right,lower = img.getbbox()
    if not (left == 0 or upper == 0 or right == current_canvas_size or left == current_canvas_size):
        break
    else:
        current_canvas_size = current_canvas_size * 1.2
        
ratio = min(size/(right-left),size/(lower - upper))
w = int(ratio * (right - left)/2)*2
h = int(ratio * (lower - upper)/2)*2
img = img.crop((left,upper,right,lower))
img = img.resize((w,h),Image.ANTIALIAS)
new_im = Image.new('L', (size,size),(0))   
new_im.paste(img, (int((size-w)/2), int((size-h)/2)))
                      
font_image = np.array(new_im)


import matplotlib.pyplot as plt

plt.imshow(font_image,interpolation="nearest",cmap='Greys')
'''
#%%
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

import os
import os.path
import random
import numpy as np


class Font:
    
    def __init__(self, size, root_dir, input_letter, output_letter):
        self.size = size
        self.input_letter = input_letter
        self.output_letter = output_letter
        
        font_files = []
        for parent,dirnames,filenames in os.walk(root_dir):  
            for filename in filenames:
                font_files.append(os.path.join(parent,filename))
        print(('Fond %i font files') % (len(font_files)))
        random.shuffle(font_files)
        self.font_files = font_files


    def getSize(self):
        return(self.size)
        
        
    def getLetterSets(self, n_train_examples, n_test_examples):
        # return a 4D numpy array that contains images of multiple letters
        
        train_input = np.zeros((n_train_examples, len(self.input_letter),self.size,self.size))
        train_output = np.zeros((n_train_examples, len(self.output_letter),self.size,self.size))
        test_input = np.zeros((n_test_examples, len(self.input_letter),self.size,self.size))
        test_output = np.zeros((n_test_examples, len(self.output_letter),self.size,self.size))
        
        m = 0
        for font_file in self.font_files[0:n_train_examples]:
            try:
                n = 0
                for letter in self.input_letter:
                    font_image = self.drawFont(font_file, letter)
                    train_input[m, n, :, :] = font_image
                    n = n + 1
                    
                n = 0
                for letter in self.output_letter:
                    font_image = self.drawFont(font_file, letter)
                    train_output[m, n, :, :] = font_image
                    n = n + 1                                        
            except:
                continue
            m = m + 1  
        train_input = train_input[0:m,:,:,:]    
        train_output = train_output[0:m,:,:,:]   
        
        m = 0
        for font_file in self.font_files[n_train_examples:n_train_examples + n_test_examples]:
            try:
                n = 0
                for letter in self.input_letter:
                    font_image = self.drawFont(font_file, letter)
                    test_input[m, n, :, :] = font_image
                    n = n + 1
                n = 0
                for letter in self.output_letter:
                    font_image = self.drawFont(font_file, letter)
                    test_output[m, n, :, :] = font_image
                    n = n + 1                    
            except:
                continue
            m = m + 1
        i = 0  
        test_input = test_input[0:m,:,:,:]
        test_output = test_output[0:m,:,:,:]

        return (train_input, train_output, test_input, test_output)
        
        
    def drawFont(self, font_file, letter):
        # draw a centered font image according to the assigned font file and letter
        font = ImageFont.truetype(font_file, self.size*2)

        current_canvas_size = self.size * 3

        while True:
            img = Image.new('L', (current_canvas_size,current_canvas_size),(0))
            draw = ImageDraw.Draw(img)
            draw.text((current_canvas_size/6,current_canvas_size/6),letter,(1),font = font)
            draw = ImageDraw.Draw(img)
            left,upper,right,lower = img.getbbox()
            if not (left == 0 or upper == 0 or right == current_canvas_size or left == current_canvas_size):
                break
            else:
                current_canvas_size = current_canvas_size * 1.2
            
        ratio = min(self.size/(right-left),self.size/(lower - upper))
        w = int(ratio * (right - left)/2)*2
        h = int(ratio * (lower - upper)/2)*2
        img = img.crop((left,upper,right,lower))
        img = img.resize((w,h),Image.ANTIALIAS)
        new_im = Image.new('L', (self.size,self.size),(0))   
        new_im.paste(img, (int((self.size-w)/2), int((self.size-h)/2)))
          
        font_image = np.array(new_im)
        
        return font_image