import cv2 as cv
import numpy as np

def remove_artifacts(image):
    # Removing artifacts
    thresh, binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # cv.imshow('Binarizado', binary)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))

    closed = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    opening = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel)

    # cv.imshow('closed', closed)
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
    # cv.imshow('erode', erode)

    contours, hierarchy = cv.findContours(erode, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    big_contour = max(contours, key = cv.contourArea)
    [height,width] = image.shape[:2]
    cv.drawContours(teste, [big_contour], 0, 255, cv.FILLED)
    # cv.imshow('teste', teste)
    dilate = cv.dilate(teste , cv.getStructuringElement(cv.MORPH_CROSS, (15, 15)), iterations=2) 
    # cv.imshow('dilate', dilate)

    removed_pectoral = cv.bitwise_and(image, 255-dilate)
    return removed_pectoral

def remove_black_space(image):
    [height,width] = image.shape[:2]
    limit = 0
    for i in range(width - 1):
        if image[20][i] != 0:
            limit = i
            break
    new = np.zeros((height, width - limit), dtype=np.uint8)
    for i in range(1024 - 1):
        for j in range(limit, width - 1):
            new[i][j-limit] = image[i][j]
    return new

def should_flip(image):
    contours, hierarchy = cv.findContours(removed_artifact, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    big_contour = max(contours, key = cv.contourArea)
    cv.drawContours(image, [big_contour], 0, (255,255, 0), 2)
    # cv.imshow('Contorno', image)
    flip = False
    for i in range(1024 - 1):
        if image[0][i] == 255:
            for j in range(500):
                if image[j][i] != 255:
                    flip = True
            break
    return flip

# for i in range(322):
selected_images = [10, 12, 15, 21, 25, 28, 58, 63, 69, 75, 83, 95, 99,
102, 115, 117, 120, 127, 132, 134, 141, 142, 144, 160, 167,
170, 175, 178, 181, 184, 199, 202, 204, 206, 207, 209,
211, 214, 218, 219, 227, 233, 238, 245, 249, 264, 267, 271,
290, 312 ]

best_fit = [15, 25, 28, 63, 83, 117, 120, 132, 134, 178, 181, 184, 202, 206, 233, 264, 267, 271]
report = [25, 28, 132, 184, 206]

for i in best_fit:
    mama = cv.imread(f'images/1 ({i}).pgm', cv.IMREAD_GRAYSCALE)
    cv.imshow(f'Imagem {i}', mama)
    cv.waitKey()

    removed_artifact = remove_artifacts(mama)
    cv.imshow('Removed Artifact', removed_artifact)
    cv.waitKey()


    if should_flip(mama):
        removed_artifact = np.fliplr(removed_artifact)
        cv.imshow('Flipada', removed_artifact)
        cv.waitKey()


    removed_black_space = remove_black_space(removed_artifact)
    # cv.imshow('Removed black space', removed_black_space)

    removed_pectoral = remove_pectoral_muscle(removed_black_space)
    cv.imshow('Musculo pectoral removido', removed_pectoral)
    cv.waitKey()

    constrasted = np.array(np.uint8(255*(removed_pectoral/255)**3.5))
    # cv.imshow('removed', constrasted)

    img = np.float32(constrasted.reshape(-1, 1))
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 500, 1.0)
    K = 5
    attempts=15
    ret,label,center=cv.kmeans(img,K,None,criteria,attempts,cv.KMEANS_PP_CENTERS)
    # print(img.shape)
    center = np.uint8(center)
    # print(label)
    # print(center)
    res = center[label.flatten()]
    segmentes = res.reshape(constrasted.shape)
    # cv.imshow('sement', segmentes)
    masked = np.copy(segmentes)
    masked[masked != max(center.flatten())] = 0
    # cv.imshow('masked', masked)
    contours, hierarchy = cv.findContours(masked, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    [height,width] = masked.shape[:2]
    cv.drawContours(removed_pectoral, contours, -1, 255, 2)
    cv.imshow('Final', removed_pectoral)


    cv.waitKey()
    cv.destroyAllWindows()
