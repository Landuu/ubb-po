import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

image_path = 'images2/photo.jpg'

def first():
    # Wczytaj obraz do analizy
    image = cv2.imread('images2/photo.jpg', cv2.IMREAD_GRAYSCALE)

    # Algorytm progowania globalnego
    _, global_thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Przykłady różnych wartości progów
    threshold_values = [50, 100, 150]

    # Inicjalizacja subplotów
    fig, axes = plt.subplots(2, len(threshold_values) + 1, figsize=(12, 4))

    # Obraz oryginalny
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Oryginał')
    # Segmentacja przy użyciu progowania globalnego
    axes[1, 0].imshow(global_thresholded, cmap='gray')
    axes[1, 0].set_title('Prog. globalne')

    # Przetestuj różne wartości progów i oceniaj jakość segmentacji
    for i, threshold_value in enumerate(threshold_values):
        _, thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        axes[0, i+1].imshow(image, cmap='gray')
        axes[0, i+1].set_title(f'Prog. {threshold_value}')
        axes[1, i+1].imshow(thresholded, cmap='gray')
        axes[1, i+1].set_title(f'Prog. {threshold_value}')

    # Ukrycie osi
    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def second():
    # Wczytaj obraz do analizy (upewnij się, że obraz zawiera obiekty o różnych kształtach)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Wykrywanie krawędzi za pomocą algorytmu Canny
    edges = cv2.Canny(image_gray, threshold1=30, threshold2=100)
    # Znajdź kontury obiektów na obrazie
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Inicjalizacja subplotów
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # Obraz oryginalny
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Oryginał')
    # Krawędzie obiektów
    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title('Krawędzie')
    # Wykrywanie obiektów o określonych kształtach
    shapes = ['Okręgi', 'Prostokąty', 'Elipsy']
    for shape_type in shapes:
        detected_objects = []
        if shape_type == 'Okręgi':
            detected_objects = [c for c in contours if len(cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)) > 8]
        elif shape_type == 'Prostokąty':
            detected_objects = [c for c in contours if len(cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)) == 4]
        elif shape_type == 'Elipsy':
            detected_objects = [c for c in contours if len(c) >= 5]
        # Tworzenie kopii obrazu i rysowanie konturów wykrytych obiektów
        image_with_contours = image.copy()
        cv2.drawContours(image_with_contours, detected_objects, -1, (0, 0, 255), 2)
        axes[2].imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'{shape_type}')
    # Ukrycie osi
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()



def third():
    # Wczytaj obraz do segmentacji
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Wczytaj oznaczenia pikseli na obrazie (maskę segmentacji)
    segmentation_mask = cv2.imread('maska_segmentacji.jpg', cv2.IMREAD_GRAYSCALE)
    # Przygotowanie danych treningowych
    height, width = image.shape
    X = np.column_stack((np.arange(height), np.arange(width)))
    y = segmentation_mask.ravel()
    # Stworzenie modelu SVM
    model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    # Trenowanie modelu
    model.fit(X, y)
    # Przewidywanie segmentacji na obrazie
    segmented_image = model.predict(X).reshape(height, width)
    # Inicjalizacja subplotów
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # Obraz oryginalny
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Oryginał')
    # Maska segmentacji (znanie poprawnych oznaczeń)
    axes[1].imshow(segmentation_mask, cmap='gray')
    axes[1].set_title('Maska segmentacji')
    # Obraz segmentowany przy użyciu modelu SVM
    axes[2].imshow(segmented_image, cmap='gray')
    axes[2].set_title('Segmentacja')
    # Ukrycie osi
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


third()