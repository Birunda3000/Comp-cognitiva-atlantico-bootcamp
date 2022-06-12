import cv2
import os
import pathlib
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle

def save_data(save_name, data):
    pickle_out = open(save_name+"-pickle.pickle", "wb")
    print('Saved as: '+save_name+"-pickle.pickle")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def expand2square(rgb_img, background_color=(0, 0, 0)):
    '''fill the with the background_color (grayscale take the mean) to have height = width'''
    pil_img = Image.fromarray(rgb_img)

    if pil_img.mode == 'L':
        background_color = int(
            (background_color[0] + background_color[1] + background_color[2])/3)

    elif pil_img.mode == 'RGB':
        pass

    else:
        raise Exception("No Grayscale or RGB identified")

    width, height = pil_img.size

    if width == height:
        return rgb_img

    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return np.asarray(result)

    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return np.asarray(result)


def read_imgs(path: str, height: int = 100, width: int = 100, resize: bool = False, mode: str = 'GRAYSCALE', squared: bool = False, square_color=(0, 0, 0), show: bool = False, cmap: str = 'gray', write: bool = False, write_path: str = 'data'):
    '''read images from path and return array (return [0|255] pixels), and a array with the files names'''
    modes = ['GRAYSCALE', 'RGB', 'UNCHANGED', '']
    images_array = []
    files_name = []
    path = pathlib.Path(path)
    all_path = os.listdir(path)
    for img_path in tqdm(all_path):
        full_path = os.path.join(path, img_path)
        if mode == 'RGB':
            img_array = cv2.imread(full_path, cv2.IMREAD_COLOR)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        elif mode == 'GRAYSCALE':
            img_array = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        elif mode == 'UNCHANGED':
            img_array = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        else:
            raise Exception(
                f"Mode must contain one of the following values: {modes}")
        files_name.append(img_path)
        if resize:
            img_array = cv2.resize(img_array, (width, height))
        if squared:
            img_array = expand2square(img_array, background_color=square_color)
        if show:
            plt.imshow(img_array, cmap=cmap)
            plt.show()
        images_array.append(img_array)

    if write:
        save_data(write_path, images_array)

    return images_array,files_name


def concatenate2_2_it(A, B):
    '''return C[i] = C[ A[i], B[i] ], for all i in range(len(A) or len(B)):'''
    if len(A) != len(B):
        raise Exception("Both arrays must be the same length")
    C = []
    for i in range(len(A)):
        C.append([A[i], B[i]])
    return C


def concatenate3_2_it(A, B, C):
    '''return D[i] = D[ A[i], B[i], C[i] ], for all i in range(len(A) or len(B) or len(B)):'''
    if len(A) != len(B) or len(A) != len(C):
        raise Exception("All arrays must be the same length")
    D = []
    for i in range(len(A)):
        D.append([A[i], B[i], C[i]])
    return D


def take_area(img):
    img = np.array(img)
    area = 0
    altura_max_index = 0
    a1 = 0
    altura_min_index = 3000
    a2 = 0
    largura_max_index = 0
    l1 = 0
    largura_min_index = 3000
    l2 = 0
    for i in range(img.shape[0]):  # linhas 1920
        for j in range(img.shape[1]):  # colunas 1080
            if (img[i][j] > 0.5):  # branco ou 0.5 ou 127
                if i > altura_max_index:
                    altura_max_index = i
                    a1 = j
                if i < altura_min_index:
                    altura_min_index = i
                    a2 = j
                if j > largura_max_index:
                    largura_max_index = j
                    l1 = i
                if j < largura_min_index:
                    largura_min_index = j
                    l2 = i
                area = area+1
    #print(f'Altura max: {altura_max_index} x {a1}')
    #print(f'Altura min: {altura_min_index} x {a2}')
    #print(f'Largura max: {l1} x {largura_max_index}')
    #print(f'Largura min: {l2} x {largura_min_index}')

    #print(f'Altura: {altura_max_index-altura_min_index}')
    #print(f'Largura: {largura_max_index-largura_min_index}')

    return [area, altura_max_index-altura_min_index, largura_max_index-largura_min_index]


def IOU(img1, img2):
    '''Metrica IOU: Se a previsão estiver completamente correta, IoU = 1. 
    Quanto menor a IoU, pior será o resultado da previsão.'''
    inter = np.logical_and(img1, img2)
    union = np.logical_or(img1, img2)
    iou_score = np.sum(inter) / np.sum(union)
    return iou_score
