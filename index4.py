import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import metrics

image_path = 'images4/img1.jpg';


def first():
    # Krok 1: Wczytaj obraz źródłowy
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Krok 2: Wykorzystaj operator Sobela do wykrywania krawędzi
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    # Krok 3: Wykorzystaj operator Canny do wykrywania krawędzi
    canny_edges = cv2.Canny(image, threshold1=100, threshold2=200)
    # Krok 4: Wykorzystaj operator Laplace'a do wykrywania krawędzi
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # Krok 5: Wyświetl wyniki wykrywania krawędzi
    plt.figure(figsize=(12, 6))
    plt.subplot(231), plt.imshow(image, cmap='gray'), plt.title('Obraz źródłowy')
    plt.subplot(232), plt.imshow(sobel_magnitude, cmap='gray'), plt.title('Operator Sobela')
    plt.subplot(233), plt.imshow(canny_edges, cmap='gray'), plt.title('Operator Canny')
    plt.subplot(234), plt.imshow(laplacian, cmap='gray'), plt.title('Operator Laplace\'a')
    plt.subplot(235), plt.imshow(sobel_x, cmap='gray'), plt.title('Operator Sobela (X)')
    plt.subplot(236), plt.imshow(sobel_y, cmap='gray'), plt.title('Operator Sobela (Y)')
    plt.show()


def second():
    # Wczytaj obraz
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Sprawdź, czy obraz został wczytany poprawnie
    if image is None:
        print("Nie udało się wczytać obrazu.")
    else:
        # Algorytm operatora Sobela
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        sobel_combined = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)
        # Algorytm operatora Canny
        canny = cv2.Canny(image, 100, 200)
        # Algorytm operatora Laplace'a
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        # Wylicz metryki jakości krawędzi
        psnr_sobel = metrics.peak_signal_noise_ratio(image, sobel_combined, data_range=1.0)
        ssim_sobel = metrics.structural_similarity(image, sobel_combined, data_range=1.0)
        psnr_canny = metrics.peak_signal_noise_ratio(image, canny, data_range=1.0)
        ssim_canny = metrics.structural_similarity(image, canny, data_range=1.0)
        psnr_laplacian = metrics.peak_signal_noise_ratio(image, laplacian, data_range=1.0)
        ssim_laplacian = metrics.structural_similarity(image, laplacian, data_range=1.0)
        # Wyświetl wyniki i metryki
        plt.subplot(2, 4, 1), plt.imshow(image, cmap='gray')
        plt.title('Obraz oryginalny'), plt.xticks([]), plt.yticks([])
        plt.subplot(2, 4, 2), plt.imshow(sobel_combined, cmap='gray')
        plt.title(f'Sobela\nPSNR: {psnr_sobel:.2f}\nSSIM: {ssim_sobel:.2f}'), plt.xticks([]), plt.yticks([])
        plt.subplot(2, 4, 3), plt.imshow(canny, cmap='gray')
        plt.title(f'Canny\nPSNR: {psnr_canny:.2f}\nSSIM: {ssim_canny:.2f}'), plt.xticks([]), plt.yticks([])
        plt.subplot(2, 4, 4), plt.imshow(laplacian, cmap='gray')
        plt.title(f'Laplace\'a\nPSNR: {psnr_laplacian:.2f}\nSSIM: {ssim_laplacian:.2f}'), plt.xticks([]),
        plt.yticks([])
        plt.show()

second()