import os
import glob
from PIL import Image

import numpy as np
import cv2

from disease_detector import DiseaseDetector


RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
PROJ_DIR: str = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
TEST_IMAGES = os.path.expanduser(f'{PROJ_DIR}/test_imgs/*.jpeg')
MODEL_PATH = os.path.expanduser(f'{PROJ_DIR}/model/plant-disease-model.pth')


def main():

    detector = DiseaseDetector(MODEL_PATH, auto_gen=True)

    # Read the test images
    imgs = []
    for img_path in glob.glob(TEST_IMAGES):
        imgs.append(Image.open(img_path))


    for img in imgs:
        pred = detector.predict(img)

        # Visualize the output
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_height = cv_img.shape[0]
        cv2.putText(cv_img, pred, (2, img_height - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, RED, 1)
        cv2.imshow(pred, cv_img)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()