# -*- coding: utf-8 -*-
# Python 3.x.x
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

DEBUG = True
if DEBUG:
    print('DEBUG mode')

class Fonts:
    # I use another class to deal multiple fonts file
    def __init__(self, root_dir, size = 50):
        # conbin the old code
        fonts = []
        for parent,dirnames,filenames in os.walk(root_dir):
            for filename in filenames:
                file_dir = os.path.join(parent,filename)
                font = Font(file_dir, size)
                fonts.append(font)

        print(('Fond %i font files') % (len(fonts)))
        random.shuffle(fonts)
        self.fonts = fonts

    def getLetterSets():
        pass


class Font:
    # init by ONE .tff file
    def __init__(self, font_file_dir, size = 50):
        self.dir = font_file_dir
        self.size = size
        self.font = ImageFont.truetype(self.dir, self.size)

    # Get img to visualize font to see if it centered cleanly.
    # It is centered
    def saveImgByRange(self, letter_range = (ord('A'), ord('E')) ):
        for i in range(letter_range[0], letter_range[1] + 1):
            # save in the current folder
            img = self.getImg(chr(i), 'RGBA', (225, 225, 225), (0,0,0))
            img.save(chr(i) + "_test.png")
            #if DEBUG:
                #print( np.array(img) )

    # get centered Image object
    def getImg(self, letter = 'A', mode = 'L', back_color = 1, font_color = 0):
        size = self.size
        font = self.font

        # Center the Font. start_point is at the cornor of the font
        w, h = font.getsize(letter)
        start_point = ( (size-w)/2, (size-h)/2 )

        # Create img, just for human to check
        img = Image.new(mode, (size, size), 1)
        draw = ImageDraw.Draw(img)
        draw.text(start_point, letter, font_color, font=font)
        draw = ImageDraw.Draw(img)

        if DEBUG:
            print('letter: ', letter)
            print('start_point: ', start_point)
            print('getsize: ', font.getsize(letter) )

        return img

    # get centered array
    def getArray(self, letter, mode = '1'):
        img = self.getImg(letter)
        np.array(img)
        if DEBUG:
            print( np.array(img) )

    # not sure what is the usge of getmask.
    def getmask(self, letter, mode = '1'):
        imgCore = self.font.getmask(letter, mode)
        arr = np.asarray(imgCore)
        if DEBUG:
            print('arr: ', arr)
        return arr

    def setSize(self, size):
        self.size = size
        font = ImageFont.truetype(self.dir, size)

    def getSize(self):
        return(self.size)

# Test
root_dir = '/Users/juby/code/FontGen/font'
file_dir = '/Users/juby/code/FontGen/font/chfont.ttf'
input_letter, output_letter = 'a', 'b'
test = Font(file_dir, 20)
test.getArray('a')
test.saveImgByRange()
#Fonts(root_dir)
