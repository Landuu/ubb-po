import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import io

def transform():
    # Wczytaj obraz
    image = cv2.imread('images2/photo.jpg', cv2.IMREAD_GRAYSCALE) 

    # Filtr dolnoprzepustowy (filtr uśredniający)
    kernel_size = 5 # Rozmiar kernela
    filtered_image_avg = cv2.blur(image, (kernel_size, kernel_size))

    # Filtr górnoprzepustowy (filtr Laplace'a)
    filtered_image_laplace = cv2.Laplacian(image, cv2.CV_64F)

    # Filtr medianowy
    kernel_size_median = 3 # Rozmiar kernela
    filtered_image_median = cv2.medianBlur(image, kernel_size_median)

    # Wyświetl obrazy
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Oryginalny obraz')
    plt.subplot(132)
    plt.imshow(filtered_image_avg, cmap='gray')
    plt.title('Filtr dolnoprzepustowy (uśredniający)')
    plt.subplot(133)
    plt.imshow(filtered_image_laplace, cmap='gray')
    plt.title('Filtr górnoprzepustowy (Laplace)')
    plt.figure(figsize=(6, 4))
    plt.imshow(filtered_image_median, cmap='gray')
    plt.title('Filtr medianowy')
    plt.show()

def analysis():
    # Wczytaj obraz do analizy
    image = cv2.imread('images2/photo.jpg', cv2.IMREAD_GRAYSCALE)
    # Tworzenie list filtrów do porównania
    filters = [
        {'name': 'Filtr uśredniający', 'kernel': np.ones((3, 3), np.float32)/9},
        {'name': 'Filtr Gaussa', 'kernel': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32)/16},
        {'name': 'Filtr Laplace\'a', 'kernel': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)}
    ]
    # Inicjalizacja subplotów
    fig, axes = plt.subplots(3, len(filters), figsize=(12, 8))

    # Przetwarzanie obrazu dla każdego filtra i ocena wpływu
    for i, filter_data in enumerate(filters):
    # Zastosowanie filtra
        filtered_image = cv2.filter2D(image, -1, filter_data['kernel'])
        # Obliczenie różnicy między oryginalnym obrazem a przefiltrowanym
        diff = cv2.absdiff(image, filtered_image)
        # Redukcja szumów
        noise_reduction = cv2.fastNlMeansDenoising(filtered_image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        # Wyostrzanie krawędzi
        sharpened_image = cv2.addWeighted(filtered_image, 1.5, image, -0.5, 0)
        # Wyświetlenie obrazów
        axes[0, i].imshow(filtered_image, cmap='gray')
        axes[0, i].set_title(filter_data['name'])
        axes[1, i].imshow(diff, cmap='gray')
        axes[1, i].set_title('Różnica')
        axes[2, i].imshow(noise_reduction, cmap='gray')
        axes[2, i].set_title('Redukcja szumów')

    # Ustawienie tytułów
    axes[0, 0].set_ylabel('Filtracja')
    axes[1, 0].set_ylabel('Różnica')
    axes[2, 0].set_ylabel('Redukcja szumów')

    # Ukrycie osi
    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def snr_test():
    # Wczytaj obraz oryginalny i obraz przefiltrowany
    original_image = cv2.imread('images2/photo.jpg', cv2.IMREAD_GRAYSCALE)
    filtered_image = cv2.medianBlur(original_image, 3)
    # Oblicz różnicę między obrazami (mapa szumu)
    noise_map = original_image - filtered_image
    # Oblicz moc sygnału i moc szumu
    signal_power = np.sum(original_image**2)
    noise_power = np.sum(noise_map**2)
    # Oblicz SNR
    snr = 10 * np.log10(signal_power / noise_power)
    print(f"SNR: {snr} dB")

    # Oblicz różnicę między obrazami
    difference_image = cv2.absdiff(original_image, filtered_image)
    # Wybierz region zainteresowania (ROI) zawierający krawędzie
    # Możesz dostosować obszar, aby zawierał krawędzie, które chcesz ocenić
    roi = difference_image[100:300, 100:300]
    # Wylicz średnią różnicę intensywności pikseli w ROI
    average_difference = np.mean(roi)
    # Wydrukuj średnią różnicę, im wyższa, tym wyraźniejsze krawędzie
    print(f"Średnia różnica: {average_difference}")
    # Alternatywnie, możesz obliczyć gradient obrazu przy użyciu operatora Sobela
    sobel_original = cv2.Sobel(original_image, cv2.CV_64F, 1, 1, ksize=5)
    sobel_filtered = cv2.Sobel(filtered_image, cv2.CV_64F, 1, 1, ksize=5)
    # Oblicz różnicę między gradientami
    sobel_difference = cv2.absdiff(sobel_original, sobel_filtered)
    # Oblicz średnią różnicę w obrazie gradientowym
    average_sobel_difference = np.mean(sobel_difference)
    print(f"Średnia różnica w obrazie gradientowym: {average_sobel_difference}")

snr_test()