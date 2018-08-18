import cv2
import numpy as np
import dlib
from imutils import face_utils
from keras.models import load_model

detector = dlib.get_frontal_face_detector()

model_simple = load_model('Simple_Autoencoder_face.h5')
model_deep = load_model('Deep_Autoencoder_face.h5')
model_conv = load_model('Convolutional_Autoencoder_face.h5')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def keras_predict(image):
    processed = keras_process_image(image)
    decoded_image_simple = model_simple.predict(processed)
    decoded_image_deep = model_deep.predict(processed)
    decoded_image_conv = model_conv.predict(np.reshape(processed, (-1, 50, 50, 3)))
    return decoded_image_simple, decoded_image_deep, decoded_image_conv


def keras_process_image(img):
    image_x = 50
    image_y = 50
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x * image_y * 3))
    img = img.astype('float32') / 255.
    return img


def adjust(decoded_image_simple, decoded_image_deep, decoded_image_conv):
    decoded_image_simple = decoded_image_simple.astype('float32') * 255.
    decoded_image_deep = decoded_image_deep.astype('float32') * 255.
    decoded_image_conv = decoded_image_conv.astype('float32') * 255.
    decoded_image_simple = np.reshape(decoded_image_simple, (50, 50, 3))
    decoded_image_deep = np.reshape(decoded_image_deep, (50, 50, 3))
    decoded_image_conv = np.reshape(decoded_image_conv, (50, 50, 3))
    return decoded_image_simple, decoded_image_deep, decoded_image_conv


def extract_face_info(img, img_rgb):
    faces = detector(img_rgb)
    x, y, w, h = 0, 0, 0, 0
    if len(faces) > 0:
        for face in faces:
            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            image = img[y:y + h, x:x + w]
            decoded_image_simple, decoded_image_deep, decoded_image_conv = keras_predict(image)
            decoded_image_simple, decoded_image_deep, decoded_image_conv = adjust(decoded_image_simple,
                                                                                  decoded_image_deep, decoded_image_conv
                                                                                  )
            cv2.imshow('SimpleNet', cv2.resize(np.array(decoded_image_simple, dtype=np.uint8), (300, 300),
                                               interpolation=cv2.INTER_AREA))
            cv2.imshow('DeepNet', cv2.resize(np.array(decoded_image_deep, dtype=np.uint8), (300, 300),
                                             interpolation=cv2.INTER_AREA))
            cv2.imshow('ConvNet', cv2.resize(np.array(decoded_image_conv, dtype=np.uint8), (300, 300),
                                             interpolation=cv2.INTER_AREA))


def recognize():
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        extract_face_info(img, img_rgb)
        cv2.imshow('Input', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


recognize()
