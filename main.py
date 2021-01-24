import warnings
warnings.filterwarnings('ignore')

import pygame
import numpy as np
import tkinter
from tkinter import messagebox
import tensorflow as tf
from tensorflow import keras

pygame.init()

window = pygame.display.set_mode((280,280))
pygame.display.set_caption("Neural network digit recognizer!")

run = True
buttonDown = False

def grid():
    g = 0
    window.fill((0,0,0))
    # for i in range(28):
    #     pygame.draw.line(window, (255,0,0), (g,0), (g,280))
    #     pygame.draw.line(window, (255,0,0), (0,g), (280,g))
    #     pygame.display.update()
    #     g+=10

def pixel():
    pixels = []
    i = 5
    while i<280:
        j=5
        while j<280:
            color = window.get_at((i,j))[0]
            if color > 0:
                pixels.append(1)
            else:
                pixels.append(0)
            j+=10
        i+=10
    return pixels

def getResult(pixels):

    (xTrain, yTrain), (xTest, yTest) = keras.datasets.mnist.load_data(path='mnist.npz')
    xTest = tf.keras.utils.normalize(xTest, axis=1)
    model = keras.models.load_model('/home/aryan/Desktop/Offline Digit Recognization/digitRecognizerModel.h5')

    xTest[0] = pixels
    p = model.predict(xTest)
    c = p[0][np.argmax(p[0])]

    print('Chance : ', c)

    return c, np.argmax(p[0])


while(run):
    for e in pygame.event.get():
        if e.type  == pygame.QUIT:
            run =  False

        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            buttonDown = True

        elif e.type == pygame.MOUSEMOTION and buttonDown:
            x1 = e.pos[0] - (e.pos[0]%10)
            y1 = e.pos[1] - (e.pos[1]%10)
            x2 = x1+10
            y2 = y1+10
            pygame.draw.rect(window, (255,255,255), (x1,y1,10,10))

        elif e.type == pygame.MOUSEBUTTONUP:
            buttonDown = False

        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 3:
            buttonDown = False
            grid()

        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_KP_ENTER or e.key == pygame.K_RETURN or e.key == pygame.K_SPACE:
                buttonDown = False
                p = np.array(pixel()).reshape((28,28)).T
                # print(p)
                root = tkinter.Tk().withdraw()
                chance, prediction = getResult(p)
                messagebox.showinfo("Result time!!", "Machine's prediction is : "+str(prediction)+f". There is {chance*100}% chance.")

        
    pygame.display.update()

grid()
pygame.quit()