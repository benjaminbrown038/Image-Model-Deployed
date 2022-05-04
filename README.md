# Sharpest Minds Project

Goal:

The ```main.py``` file:

1. Scrape images of cats
2. Scrape images of dogs
3. Extract faces from Images
4. Save images in folder specific to ```cat``` or ```dog``` inside ```Images``` folder

The ```data.py``` file:

1. Create augmented images
2. save 80% of augmented images to train folder
3. save 20% of augmented images to test folder

Set up:

1. If not already installed:
https://www.anaconda.com/products/individual
2. Create the virtual environment.
Run: ```conda create -n envname python=3.8 anaconda ```
3. Clone
Run: ``` git clone https://www.github.com/benjaminbrown038/Sharpest-Minds-Project.git```
4. Activate:
```conda activate envname```
5. Install packages:
```pip install requirements.txt ```
6. Run program:
```python3 main.py```

This will create a folder with the class of images. 

Notes:
  - Chromedrivers can be found at https://chromedriver.chromium.org/downloads.

References:

  - https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html.

  https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/

