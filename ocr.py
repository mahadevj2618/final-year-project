import easyocr as ea
import cv2
import pathlib
import numpy as np

from matplotlib import pyplot as plt


def perform_ocr(filepath):
    reader = ea.Reader(['mr'])
    result = reader.readtext(filepath)

    img = cv2.imread(filepath)

    text1 = ''
    for detection in result:
        top_left = tuple([int(val) for val in detection[0][0]])
        bottom_right = tuple([int(val) for val in detection[0][2]])
        text = detection[1]
        text1 += text

        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.rectangle(img, top_left, bottom_right, (0, 200, 0), 5)

        img = cv2.putText(img, text, top_left, font, 1, (0, 0, 200), 2, cv2.LINE_AA)

    with open("output.txt", "w", encoding="utf-8") as file:
        # Write the text to the file
        file.write(str(text1))


    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    # plt.axis('off')
    plt.show()

    return result

