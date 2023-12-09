import cv2
import numpy as np 
import matplotlib.pyplot as plt

def convert_color_spaces(image):
    """Konwersja obrazu do róznych prestrzeni kolorów."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return rgb_image, hsv_image, lab_image

def white_balance_correction(image):
    """Korekcja balansu bieli obrazu."""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    l_channel_avg = np.mean(l_channel)
    a_channel = a_channel - ((l_channel_avg - 128) * (a_channel.std() / 128.0))
    b_channel = b_channel - ((l_channel_avg - 128) * (b_channel.std() / 128.0))
    # Skalowanie Ranatow do zakresu 0-255
    l_channel = cv2.normalize(l_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    a_channel = cv2.normalize(a_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    b_channel = cv2.normalize(b_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # Połączenie kanałów
    corrected_lab_image = cv2.merge([l_channel.astype(np.uint8), a_channel.astype(np.uint8), b_channel.astype(np.uint8)])
    corrected_image = cv2.cvtColor(corrected_lab_image, cv2.COLOR_LAB2BGR)
    return corrected_image

def remove_red_eye_effect(image):
    """Usuwanie efektu czerwonych oczu z obrazu."""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    red_mask = cv2.inRange(b_channel, 150, 255)
    eye_mask = red_mask > 0
    l_channel[eye_mask] = np.mean(l_channel)
    l_channel = cv2.resize(l_channel, (b_channel.shape[1], b_channel.shape[0]))
    corrected_lab_image = cv2.merge([l_channel, a_channel, b_channel])
    corrected_image = cv2.cvtColor(corrected_lab_image, cv2.COLOR_LAB2BGR)
    return corrected_image

image_path_color_space = "images5/colors.jpg"
image_path_white_balance = "images5/balance.jpg"
image_path_red_eye = "images5/redeye.jpg"

def ex1():
    image_color_space = cv2.imread(image_path_color_space)
    image_white_balance = cv2.imread(image_path_white_balance)
    image_red_eye = cv2.imread(image_path_red_eye)

    # Zad 1
    rgb_image, hsv_image, lab_image = convert_color_spaces(image_color_space)
    # Zad 2
    corrected_white_balance_image = white_balance_correction(image_white_balance)
    # Zad 3
    corrected_red_eye_image = remove_red_eye_effect(image_red_eye)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(image_color_space, cv2.COLOR_BGR2RGB))
    plt.title("Oryginalny obraz")
    plt.axis("off")
    plt.subplot(1, 4, 2)
    plt.imshow(hsv_image)
    plt.title("Obraz HSV")
    plt.axis("off")
    plt.subplot(1, 4, 3)
    plt.imshow(corrected_white_balance_image)
    plt.title("Korekcja balansu bieli")
    plt.axis("off")
    plt.subplot(1, 4, 4)
    plt.imshow(corrected_red_eye_image)
    plt.title("Korekcja czerwonych oczu")
    plt.axis("off")
    plt.show()


def ex2():
    # Wczytaj obraz kolorowy
    original_image = cv2.imread(image_path_color_space)
    # Konwersja z BGR (domyślna przestrzeń OpenCV) do RGB
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # Konwersja z BGR do HSV
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    # Wyświetl obraz oryginalny, RGB i HSV obok siebie
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Oryginalny obraz OpenCV (BGR)")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(rgb_image)
    plt.title("RGB Image")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(hsv_image)
    plt.title("HSV Image")
    plt.axis("off")
    plt.show()

def ex_bb():
    image_white_balance = cv2.imread(image_path_white_balance)
    corrected_white_balance_image = white_balance_correction(image_white_balance)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_white_balance)
    plt.title("Oryginalny obraz")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(corrected_white_balance_image)
    plt.title("Korekcja balansu bieli")
    plt.axis("off")
    plt.show()

def ex_bo():
    image_red_eye = cv2.imread(image_path_red_eye)
    corrected_red_eye_image = remove_red_eye_effect(image_red_eye)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_red_eye)
    plt.title("Oryginalny obraz")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(corrected_red_eye_image)
    plt.title("Korekcja czerwonych oczu")
    plt.axis("off")
    plt.show()

ex_bo()
