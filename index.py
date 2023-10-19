from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2 as cv

src1 = cv.imread('images/klasa1.jpg')
src1 = cv.cvtColor(src1, cv.COLOR_BGR2GRAY)
dst1 = cv.equalizeHist(src1)

psnr_value = psnr(src1, dst1)
print(psnr_value)