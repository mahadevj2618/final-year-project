import pathlib

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.models import load_model
import numpy as np


def test_model(image_path):
    batch_size = 32
    img_height = 180
    img_width = 180

    class_names = ['C1', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C2', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C3', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38', 'C39', 'C4', 'C40', 'C41', 'C42', 'C43', 'C44', 'C45', 'C46', 'C47', 'C48', 'C49', 'C5', 'C50', 'C51', 'C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58', 'C6', 'C7', 'C8', 'C9']
    result = {'C1': 'अ', 'C2': 'आ', 'C3': 'इ', 'C4': 'ई', 'C5': 'उ', 'C6': 'ऊ', 'C7': 'ए', 'C8': 'ऐ', 'C9': 'ओ', 'C10': 'औ', 'C11': 'अं', 'C12': 'अः', 'C13': 'क', 'C14': 'ख', 'C15': 'ग', 'C16': 'घ', 'C17': 'ङ', 'C18': 'च', 'C19': 'छ', 'C20': 'ज', 'C21': 'झ', 'C22': 'ञ', 'C23': 'ट', 'C24': 'ठ', 'C25': 'ड', 'C26': 'ढ', 'C27': 'ण', 'C28': 'त', 'C29': 'थ', 'C30': 'द', 'C31': 'ध', 'C32': 'न', 'C33': 'प', 'C34': 'फ', 'C35': 'ब', 'C36': 'भ', 'C37': 'म', 'C38': 'य', 'C39': 'र', 'C40': 'ल', 'C41': 'व', 'C42': 'श', 'C43': 'ष', 'C44': 'स', 'C45': 'ह', 'C46': 'ळ', 'C47': 'क्ष', 'C48': 'ज्ञ', 'C49': '०', 'C50': '१', 'C51': '२', 'C52': '३', 'C53': '४', 'C54': '५', 'C55': '६', 'C56': '७', 'C57': '८', 'C58': '९'}

    num_classes = len(class_names)

    model = load_model('detect_image.h5')

    # data_dir = pathlib.Path("C:/Users/Dell/Downloads/Mar.jpg")
    data_dir = pathlib.Path(image_path)

    img = tf.keras.utils.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    rs = class_names[np.argmax(score)]
    print("category :", class_names[np.argmax(score)])
    text=result[rs]
    with open("output.txt", "w", encoding="utf-8") as file:
        # Write the text to the file
        file.write(str(text))

    return result[rs]
