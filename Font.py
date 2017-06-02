# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

import os
import os.path
import random
import numpy as np

import Parameters as params

from matplotlib import pyplot as plt

class Font:
    
    def __init__(self):
        self.size = params.basis_size
        self.input_letter = params.input_letter
        self.output_letter = params.output_letter
        
        font_files = []
        for parent,dirnames,filenames in os.walk(params.font_dir):  
            for filename in filenames:
                font_files.append(os.path.join(parent,filename))
        print(('Fond %i font files') % (len(font_files)))
        random.shuffle(font_files)
        self.font_files = font_files


    def getSize(self):
        return(self.size)
        
        
    def getLetterSets(self):
        # return a 4D numpy array that contains images of multiple letters
        total_examples = len(self.font_files)
        n_test_examples = int(total_examples / params.TRAIN_TEST_SPLIT)
        n_train_examples = total_examples - n_test_examples
        train_idx = np.random.choice(total_examples, size = n_train_examples, replace=False)
        test_idx = np.setdiff1d(list(range(total_examples)), train_idx)
        
        all_input = np.zeros((total_examples, self.size, self.size, len(self.input_letter)))
        all_output = np.zeros((total_examples, self.size, self.size, len(self.output_letter)))        
        for idx, font_file in enumerate(self.font_files):
            try:
                for letter_idx, letter in enumerate(self.input_letter):
                    font_image = self.drawFont(font_file, letter)
                    all_input[idx, :, :, letter_idx] = font_image
                    
                for letter_idx, letter in enumerate(self.output_letter):
                    font_image = self.drawFont(font_file, letter)
                    all_output[idx, :, :, letter_idx] = font_image        
            except:
                continue                               
        
        train_input = all_input[train_idx]
        train_output = all_output[train_idx]
        test_input = all_input[test_idx]
        test_output = all_output[test_idx]

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
                current_canvas_size = int(current_canvas_size * 1.2)
            
        ratio = min(self.size/(right-left),self.size/(lower - upper))
        w = int(ratio * (right - left)/2)*2
        h = int(ratio * (lower - upper)/2)*2
        img = img.crop((left,upper,right,lower))
        img = img.resize((w,h),Image.ANTIALIAS)
        new_im = Image.new('L', (self.size,self.size),(0))   
        new_im.paste(img, (int((self.size-w)/2), int((self.size-h)/2)))
          
        font_image = np.array(new_im)
        
        return font_image
        
def test_display(predicted, truth, expIdx):
    width = predicted.shape[3]
    plt.figure(1)
    plt.axis('off')
    for i in range(width):
        plt.subplot(2,width,i+1)
        plt.imshow(predicted[expIdx, :, :, i-1],interpolation="nearest",cmap='Greys')
        plt.subplot(2,width,i+width+1)
        plt.imshow(truth[expIdx, :, :, i-1],interpolation="nearest",cmap='Greys')
    plt.show()
    plt.pause(0.0001) 