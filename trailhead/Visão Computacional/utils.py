import matplotlib.pyplot as plt
import cv2

def show_img(img):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        pass
    plt.imshow(img, cmap='gray')
    plt.show()