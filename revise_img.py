from PIL import Image

import os

def img_resize(file, h, w):
  img=Image.open(file)
  Resize = img.resize((h,w), Image.ANTIALIAS) 
  Resize.save('resized.jpg', 'JPEG', quality=90) 
img＿resize（＂文件路径＂，400,200）
