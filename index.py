import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import io

# Ustawienia
class1_path = 'images/klasa1.jpg'
class2_path = 'images/klasa2.jpg'
class3_path = 'images/klasa3.jpg'
class4_path = 'images/klasa4.jpg'





#
# 1. Porównianie oryginalnych obrazów
#

# Wczytywanie obrazów
image1 = cv2.imread(class1_path, cv2.IMREAD_GRAYSCALE) 
image2 = cv2.imread(class2_path, cv2.IMREAD_GRAYSCALE) 
image3 = cv2.imread(class3_path, cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread(class4_path, cv2.IMREAD_GRAYSCALE)

# Obliczanie histogramów
histogram1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
histogram2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
histogram3 = cv2.calcHist([image3], [0], None, [256], [0, 256])
histogram4 = cv2.calcHist([image4], [0], None, [256], [0, 256])

def show_histogram_plot(img_number, original_image):
    histogram = cv2.calcHist([original_image], [0], None, [256], [0, 256])
    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(original_image, cmap='gray'), plt.title(f'Obraz {img_number}')
    plt.subplot(132), plt.plot(histogram), plt.title(f'Histogram Obrazu {img_number}')
    plt.subplot(133), plt.hist(original_image.ravel(), 256, [0, 256]), plt.title(f'Histogram Obrazu {img_number} (inna metoda)')
    plt.show()

# Wyświetlenie histogramów
show_histogram_plot(1, image1)
show_histogram_plot(2, image2)
show_histogram_plot(3, image3)
show_histogram_plot(4, image4)






#
# 2. Transformacje obrazów
# 

# Wyrównanie histogramu (EQUALIZATION)
equalized_image1 = cv2.equalizeHist(image1)
equalized_image2 = cv2.equalizeHist(image2)
equalized_image3 = cv2.equalizeHist(image3)
equalized_image4 = cv2.equalizeHist(image4)

# Rozciąganie histogramu (STRECHING)
min_pixel_value = 50 # Minimalna wartość piksela po rozciągnięciu
max_pixel_value = 200 # Maksymalna wartość piksela po rozciągnięciu
stretched_image1 = cv2.normalize(image1, None, min_pixel_value, max_pixel_value, cv2.NORM_MINMAX)
stretched_image2 = cv2.normalize(image2, None, min_pixel_value, max_pixel_value, cv2.NORM_MINMAX)
stretched_image3 = cv2.normalize(image3, None, min_pixel_value, max_pixel_value, cv2.NORM_MINMAX)
stretched_image4 = cv2.normalize(image4, None, min_pixel_value, max_pixel_value, cv2.NORM_MINMAX)

# Skalowanie histogramu (SCALING)
alpha = 1.5 # Współczynnik skali
scaled_image1 = cv2.multiply(image1, alpha)
scaled_image2 = cv2.multiply(image2, alpha)
scaled_image3 = cv2.multiply(image3, alpha)
scaled_image4 = cv2.multiply(image4, alpha)

# Wygładzanie i wyostrzanie (za pomocą filtra Gaussa)
kernel_size = (5, 5) # Rozmiar jądra filtra Gaussa
smoothed_image1 = cv2.GaussianBlur(image1, kernel_size, 0)
smoothed_image2 = cv2.GaussianBlur(image2, kernel_size, 0)
smoothed_image3 = cv2.GaussianBlur(image3, kernel_size, 0)
smoothed_image4 = cv2.GaussianBlur(image4, kernel_size, 0)
sharpened_image1 = cv2.addWeighted(image1, 1.5, smoothed_image1, -0.5, 0)
sharpened_image2 = cv2.addWeighted(image2, 1.5, smoothed_image2, -0.5, 0)
sharpened_image3 = cv2.addWeighted(image3, 1.5, smoothed_image3, -0.5, 0)
sharpened_image4 = cv2.addWeighted(image4, 1.5, smoothed_image4, -0.5, 0)

def show_transformed_plot(img_number, original_image, equalized_image, stretched_image, scaled_image, smoothed_image, sharpened_image):
    plt.figure(figsize=(12, 4))
    plt.subplot(231), plt.imshow(original_image, cmap='gray'), plt.title(f'Obraz {img_number}')
    plt.subplot(232), plt.imshow(equalized_image, cmap='gray'), plt.title(f'Wyrównany Obraz {img_number}')
    plt.subplot(233), plt.imshow(stretched_image, cmap='gray'), plt.title(f'Rozciągnięty Obraz {img_number}')
    plt.subplot(234), plt.imshow(scaled_image, cmap='gray'), plt.title(f'Skalowany Obraz {img_number}')
    plt.subplot(235), plt.imshow(smoothed_image, cmap='gray'), plt.title(f'Wygładzony Obraz {img_number}')
    plt.subplot(236), plt.imshow(sharpened_image, cmap='gray'), plt.title(f'Wyostrzony Obraz {img_number}')
    plt.show()

# Wyświetlenie obrazów wynikowych
show_transformed_plot(1, image1, equalized_image1, stretched_image1, scaled_image1, smoothed_image1, sharpened_image1)
show_transformed_plot(2, image2, equalized_image2, stretched_image2, scaled_image2, smoothed_image2, sharpened_image2)
show_transformed_plot(3, image3, equalized_image3, stretched_image3, scaled_image3, smoothed_image3, sharpened_image3)
show_transformed_plot(4, image4, equalized_image4, stretched_image4, scaled_image4, smoothed_image4, sharpened_image4)





#
# 3. Ocena PSNR
#

def show_psnr_ssim(img_number, original_image, equalized_image, stretched_image, scaled_image, smoothed_image, sharpened_image):
    # Obliczenie wskaźnika PSNR
    psnr_equalized = psnr(original_image, equalized_image)
    psnr_stretched = psnr(original_image, stretched_image)
    psnr_scaled = psnr(original_image, scaled_image)
    psnr_smoothed = psnr(original_image, smoothed_image)
    psnr_sharpened = psnr(original_image, sharpened_image)
    
    # Obliczenie wskaźnika SSIM
    ssim_equalized = ssim(original_image, equalized_image)
    ssim_stretched = ssim(original_image, stretched_image)
    ssim_scaled = ssim(original_image, scaled_image)
    ssim_smoothed = ssim(original_image, smoothed_image)
    ssim_sharpened = ssim(original_image, sharpened_image)
    
    # Wyświetlenie wyników
    print(f'-------- OBRAZ NR {img_number} --------')
    print(f'PSNR po wyrównaniu histogramu: {psnr_equalized:.2f}')
    print(f'PSNR po rozciągnięciu histogramu: {psnr_stretched:.2f}')
    print(f'PSNR po skalowaniu histogramu: {psnr_scaled:.2f}')
    print(f'PSNR po wygładzeniu histogramu: {psnr_smoothed:.2f}')
    print(f'PSNR po wyostrzeniu histogramu: {psnr_sharpened:.2f}')

    print(f'SSIM po wyrównaniu histogramu: {ssim_equalized:.2f}')
    print(f'SSIM po rozciągnięciu histogramu: {ssim_stretched:.2f}')
    print(f'SSIM po skalowaniu histogramu: {ssim_scaled:.2f}')
    print(f'SSIM po wygładzeniu histogramu: {ssim_smoothed:.2f}')
    print(f'SSIM po wyostrzeniu histogramu: {ssim_sharpened:.2f}')
    print('')

show_psnr_ssim(1, image1, equalized_image1, stretched_image1, scaled_image1, smoothed_image1, sharpened_image1)
show_psnr_ssim(2, image2, equalized_image2, stretched_image2, scaled_image2, smoothed_image2, sharpened_image2)
show_psnr_ssim(3, image3, equalized_image3, stretched_image3, scaled_image3, smoothed_image3, sharpened_image3)
show_psnr_ssim(4, image4, equalized_image4, stretched_image4, scaled_image4, smoothed_image4, sharpened_image4)
