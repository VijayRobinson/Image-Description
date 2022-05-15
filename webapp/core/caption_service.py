import cv2
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50

from keras.preprocessing.sequence import pad_sequences
from numpy import array
import numpy as np
import json
import keras

from webapp.settings import I2W_PATH, MODEL_PATH, W2I_PATH


class CaptionService():
    def __init__(self):
        self.model = keras.models.load_model(MODEL_PATH)
        self.max_length = 40
        self.res_model = ResNet50(
            include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3))
        self.words_to_indices = json.load(open(W2I_PATH, 'r'))
        self.indices_to_words = json.load(open(I2W_PATH, 'r'))

    def get_caption(self, image_path):
        photo = self.get_features(image_path)
        caption = self.greedy_search(photo)
        return " ".join(caption)

    def get_features(self, image_path):
        image = np.asarray(image_path)
        x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = cv2.resize(image, (224, 224))
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = self.res_model.predict(x)
        return features.squeeze()

    def greedy_search(self, photo):
        photo = photo.reshape(1, 2048)
        in_text = '<start>'
        for i in range(self.max_length):
            sequence = [self.words_to_indices[s]
                        for s in in_text.split(" ") if s in self.words_to_indices]
            sequence = pad_sequences(
                [sequence], maxlen=self.max_length, padding='post')
            y_pred = self.model.predict([photo, sequence], verbose=0)
            y_pred = np.argmax(y_pred[0])
            word = self.indices_to_words[str(y_pred)]
            in_text += ' ' + word
            if word == '<end>':
                break
        final = in_text.split()
        final = final[1:-1]
        return final
