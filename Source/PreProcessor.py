import PIL
from PIL import Image
import os

def resize_image (img):
    
    wsize = img.size[0]/2
    hsize = img.size[1]/2
    img_resized = img.resize((wsize,hsize), PIL.Image.ANTIALIAS)
    
    return img_resized

def crop_image (img, x, y, w, h):
    img_cropped = img.crop((x, y, w+x, h+y))

    return img_cropped


def adjust_image_size (img):
    wsize = 511
    hsize = 384
    img_resized = img.resize((wsize,hsize), PIL.Image.ANTIALIAS)
    
    return img_resized


def save_image (img, destination):
        img.save(destination)
        img.close()
        
if __name__ == "__main__":
    source = 'Original_Data'
    destination = 'Cropped_Resized_Data'

    x = 0
    y = (1272/2)
    w = (3680/2)
    h = (500/2)

    for i in range(len(os.listdir(source))):
        path_s = os.listdir(source)
        path_d = range(len(path_s))
        path_d = os.path.join(destination, path_s[i])
        path_s = os.path.join(source, path_s[i])

        if (os.path.isfile(path_s)):
           files = path_s
        if (files.lower().endswith('.jpg')):
            jpgs = files
        img = PIL.Image.open(jpgs)
        
        img = resize_image(img)
        
        img = crop_image(img, x, y, w, h)
    
        img = adjust_image_size(img)

        save_image(img, path_d)

