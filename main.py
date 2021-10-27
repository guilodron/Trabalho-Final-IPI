from os import remove
import cv2 as cv
import numpy as np

def remove_artifacts(image):
    # Removing artifacts
    thresh, binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # cv.imshow('Binarizado', binary)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))

    closed = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    opening = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel)
    # cv.imshow('Abertura', opening)

    contours, hierarchy = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    big_contour = max(contours, key = cv.contourArea)
    [height,width] = image.shape[:2]
    mask = np.zeros((height,width), dtype=np.uint8)
    cv.drawContours(mask, [big_contour], 0, 255, cv.FILLED)
    # cv.imshow('Mascara contorno', mask)

    maskDilated = cv.dilate(mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (200, 200)), 100)
    # cv.imshow('Mascara final', maskDilated)
    removedArtifact = cv.bitwise_and(maskDilated, image)
    return removedArtifact

def remove_pectoral_muscle(image):
    vals = []
    for i in range(50):
        for j in range(50):
            vals.append(image[i][j])
    limite = np.mean(vals)
    # print('limite', limite)
    [height,width] = image.shape[:2]
    teste = np.zeros((height,width), dtype=np.uint8)
    # limite = np.mean(image) + np.std(image) * 1.2
    # print('limite', limite)
    thresh, binary = cv.threshold(image, thresh=limite*0.8, maxval=255, type=cv.THRESH_BINARY)
    # cv.imshow('binary', binary)
    erode = cv.erode(binary,cv.getStructuringElement(cv.MORPH_RECT, (10, 10)), iterations=2 )
    # dilate = cv.dilate(binary , cv.getStructuringElement(cv.MORPH_RECT, (10, 10)), iterations=2) 
    contours, hierarchy = cv.findContours(erode, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    big_contour = max(contours, key = cv.contourArea)
    [height,width] = image.shape[:2]
    cv.drawContours(teste, [big_contour], 0, 255, cv.FILLED)

    removed_pectoral = cv.bitwise_and(image, 255-teste)
    return removed_pectoral

def remove_black_space(image):
    [height,width] = image.shape[:2]
    limit = 0
    for i in range(width - 1):
        if image[20][i] != 0:
            limit = i
            break
    print(limit)
    new = np.zeros((height, width - limit), dtype=np.uint8)
    print('new', new.shape)
    for i in range(1024 - 1):
        for j in range(limit, width - 1):
            new[i][j-limit] = image[i][j]
    return new

def should_flip(image):
    contours, hierarchy = cv.findContours(removed_artifact, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    big_contour = max(contours, key = cv.contourArea)
    cv.drawContours(image, [big_contour], 0, (255,255, 0), 2)
    flip = False
    for i in range(1024 - 1):
        if image[0][i] == 255:
            for j in range(500):
                if image[j][i] != 255:
                    flip = True
            break
    return flip

for i in range(322):
    mama = cv.imread(f'images/1 (184).pgm', cv.IMREAD_GRAYSCALE)
    cv.imshow('Original', mama)
    removed_artifact = remove_artifacts(mama)

    if should_flip(mama):
        removed_artifact = np.fliplr(removed_artifact)

    removed_black_space = remove_black_space(removed_artifact)
    # cv.imshow('Removed black space', removed_black_space)



    removed_pectoral = remove_pectoral_muscle(removed_black_space)
    # cv.imshow('Musculo pectoral removido', removed_pectoral)

    # clahe = cv.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    # equalized = clahe.apply(removed_pectoral)

    # equalized = np.array(np.uint8(255*(removed_pectoral/255)**2))
    # cv.imshow('equalized', equalized)


    # kernel_8 = np.array([[1,1,1], [1, -8, 1], [1, 1, 1]])
    # laplacian = cv.filter2D(equalized, -1, kernel_8)
    # sharpened = cv.subtract(equalized, laplacian)
    # cv.imshow('Sharpened', sharpened)

    # trhesh, binary = cv.threshold(sharpened, thresh=80, maxval=255, type=cv.THRESH_BINARY)

    constrasted = np.array(np.uint8(255*(removed_pectoral/255)**3.5))
    # cv.imshow('removed', constrasted)

    img = np.float32(constrasted.reshape(-1, 1))
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 500, 1.0)
    K = 3
    attempts=10
    ret,label,center=cv.kmeans(img,K,None,criteria,attempts,cv.KMEANS_PP_CENTERS)
    print(img.shape)
    center = np.uint8(center)
    print(label)
    print(center)
    res = center[label.flatten()]
    segmentes = res.reshape(constrasted.shape)
    cv.imshow('sement', segmentes)
    masked = np.copy(segmentes)
    masked[masked != max(center.flatten())] = 0
    # cv.imshow('masked', masked)
    contours, hierarchy = cv.findContours(masked, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    [height,width] = masked.shape[:2]
    cv.drawContours(removed_pectoral, contours, -1, 255, 2)
    cv.imshow('Final', removed_pectoral)


    cv.waitKey()
    cv.destroyAllWindows()


# contrasted = np.array(np.uint8(255*(removed_pectoral/255)**5))
# cv.imshow('contrasted', contrasted)





# contours, hierarchy = cv.findContours(removed_artifact, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# big_contour = max(contours, key = cv.contourArea)
# cv.drawContours(mask, [big_contour], 0, 255, 2)
# cv.imshow('Contornos', mask)




# Removendo músculo pectorial










cv.waitKey()