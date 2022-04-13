import cv2
import numpy as np
import image_processing
from tensorflow.keras.models import load_model

# Colab Link for model - https://colab.research.google.com/drive/1QY63u9PivDM2QQ_-qeEX4UlQpFw_tXTK?usp=sharing

# Dictionary to decode Labels
dict = {0: 1,1: 2,2: 3, 3: 4,4: 5,5: 6,6: 7,7: 10,8: 13,9: 14,10: 15,11: 17,12: 23,13: 25,14: 27,15: 28,16: 29,17: 30,18: 35,19: 36,20: 38,21: 39,22: 42,23: 43,24: 45,25: 46}

# Dictionary to print the character
word_dict = {1: '1',2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',10:'A',13:'D',14:'E',15:'F',17:'H',23:'N',25:'P',27:'R',28:'S',29:'T',30:'U',35:'Z', 36: 'a',38:'d',39:'e',42:'h',43:'n',45:'r',46:'t'}

def predict(img):
    ans = []

    # RESIZING
    img = cv2.resize(img, (500, 500))

    # REMOVE SHADOW
    img = image_processing.remove_shadow(img)

    # CONVERTING TO GRAYSCALE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # APPLYING GAUSSIAN BLUR
    gaussian_blur = cv2.GaussianBlur(gray, (3,3), sigmaX = 0)

    # CREATING THRESHOLD
    _, thresh = cv2.threshold(gaussian_blur, 220, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # REMOVING NOISE
    img2 = image_processing.remove_noise(thresh)

    kernel2 = np.ones((1, 1), dtype=np.uint8)
    img2 = cv2.dilate(img2, kernel2, iterations = 1)

    # CREATING COPY
    img3 = img2.copy()

    # SEGMENTATION
    segmented_img = image_processing.segmentation(img2, img3)
    segmented_img = image_processing.purify(segmented_img)


    # CROPING 
    roi_segmeted_img = image_processing.crop(segmented_img, img2)

    # LOADING MODEL
    model_path = "model.h5"
    model = load_model(model_path)

    # DISPLAYING SEGMENTED IMAGES
    for i, img in enumerate(roi_segmeted_img):
        img_inv = cv2.bitwise_not(img)
        kernal = np.ones((2, 2), dtype=np.uint8)
        img_inv = cv2.erode(img_inv, kernal, iterations=1)
        image_processing.display(img_inv, "img " +  str(i))

    # PREDICTING CHARACTER
    for img in roi_segmeted_img:
        img_inv = cv2.bitwise_not(img)
        kernal = np.ones((2, 2), dtype=np.uint8)
        img_inv = cv2.erode(img_inv, kernal, iterations=1)
        img_inv = np.reshape(img_inv,(1,28,28,1))
        pred = model.predict(img_inv)
        ans.append([dict[np.argmax(pred)]])

    return ans


def test():

    img_paths = ["Sample Images/sample1.jpeg", "Sample Images/sample2.jpg"]
    
    for i, path in enumerate(img_paths):
        # READING IMAGE
        img = cv2.imread(path)
        ans = predict(img)
        print("Captcha " + str(i+1) +":")
        for i in ans:
            print(word_dict[i[0]], end =" ")
        print('\n')


if __name__ == "__main__":
    test()

