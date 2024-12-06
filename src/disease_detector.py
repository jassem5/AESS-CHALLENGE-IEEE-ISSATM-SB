import os

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

from resnet_9 import ResNet9

from typing import Optional

class DiseaseDetector:
    '''
    Wrapper class for the ResNet9 Classifier model that's trained on the plant disease
    dataset. This wrapper aims at simplifying the interface with the ML model.
    '''

    CLASSES = [
        'Apple: Apple scab',
        'Apple: Black rot',
        'Apple: Cedar apple rust',
        'Apple: healthy',
        'Blueberry: healthy',
        'Cherry[including sour]: Powdery mildew',
        'Cherry[including sour]: healthy',
        'Corn[maize]: Gray leaf spot',
        'Corn[maize]: Common rust ',
        'Corn[maize]: Northern Leaf Blight',
        'Corn[maize]: healthy',
        'Grape: Black rot',
        'Grape: Esca (Black Measles)',
        'Grape: Leaf blight (Isariopsis Leaf Spot)',
        'Grape: healthy',
        'Orange: Haunglongbing (Citrus greening)',
        'Peach: Bacterial spot',
        'Peach: healthy',
        'Pepper[bell]: Bacterial spot',
        'Pepper[bell]: healthy',
        'Potato: Early blight',
        'Potato: Late blight',
        'Potato: healthy',
        'Raspberry: healthy',
        'Soybean: healthy',
        'Squash: Powdery mildew',
        'Strawberry: Leaf scorch',
        'Strawberry: healthy',
        'Tomato: Bacterial spot',
        'Tomato: Early blight',
        'Tomato: Late blight',
        'Tomato: Leaf Mold',
        'Tomato: Septoria leaf spot',
        'Tomato: Spider mites',
        'Tomato: Target Spot',
        'Tomato: Tomato Yellow Leaf Curl Virus',
        'Tomato: Tomato mosaic virus',
        'Tomato: healthy'
    ]
    NUM_CLASSES: int = len(CLASSES)
    NUM_INPUTS: int = 3
    QUANT_PREFIX: str = 'quantized'
    PROJ_DIR: str = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
    DEFAULT_CALIB_DATA_PATH: str = f'{PROJ_DIR}/data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid'
    BACKEND: str = 'fbgemm'  # 'x86', default: 'fbgemm'

    def __init__(self, model_path: str, *,
                 quantized: bool = True,
                 auto_gen: bool = True,
                 calib_data_path: Optional[str] = None
                 ) -> None:
        '''
        Read in the saved model, and prepare if for use
        '''
        self._model_path = model_path
        self._quant_model_path = f"{self._model_path.split('.')[0]}__{self.QUANT_PREFIX}.pth"
        quant_model_exists: bool = os.path.exists(self._quant_model_path)
        if quantized and not quant_model_exists and not auto_gen:
            assert False, f'Quantized model does not exit at: {self._quant_model_path}, and auto_gen is False.'

        self._model = ResNet9(self.NUM_INPUTS, self.NUM_CLASSES)
        self._model.load_state_dict(torch.load(model_path))
        self._model.eval()

        if quantized and quant_model_exists:
            self.__load_quantized_model()
        elif quantized and not quant_model_exists and auto_gen:
            print("QUANTIZED MODEL NOT FOUND, GOING TO GENERATE IT!")
            self.__quantize(calib_data_path or self.DEFAULT_CALIB_DATA_PATH)
            self.__load_quantized_model()

        # Create a transformation function that would turn images into tensors
        self._transform = transforms.ToTensor()

    def predict(self, img):
        '''
        Use the model to predict the input image
        '''
        input_tensor = self.__preprocess(img)
        yb = self._model(input_tensor)
        _, pred = torch.max(yb, dim=1)
        res = self.__postprocess(pred[0].item())
        return res

    def __preprocess(self, img) -> torch.Tensor:
        '''
        Apply any transformations that are necessary to the input image in order to make it
        in the form that the model expects
        (in our case, all we need to do is make it into a Tensor, but we might want to check
        if it's the correct shape, and make it into the correct one if it's not ...)
        '''
        tensor_from_img = self._transform(img).unsqueeze(0)
        return tensor_from_img

    def __postprocess(self, pred: int) -> str:
        '''
        Use the output of the model and interpret in any way suits you, in our case,
        we simply return the corresponding label to the output index
        '''
        return self.CLASSES[pred]

    def __load_quantized_model(self):
        quantized_model = ResNet9(self.NUM_INPUTS, self.NUM_CLASSES)
        quantized_model.qconfig = torch.quantization.get_default_qconfig(self.BACKEND)
        quantized_model = torch.quantization.prepare(quantized_model)
        quantized_model = torch.quantization.convert(quantized_model)
        quantized_model.load_state_dict(torch.load(self._quant_model_path), strict=False)
        quantized_model.eval()
        self._model = quantized_model

    def __quantize(self, calib_data_path: str):
        '''
        Converts the model into a quantized one using the provided calibration data
        '''
        fuse_modules_list = [
            ['conv1.0', 'conv1.1', 'conv1.2'],
            ['conv2.0', 'conv2.1', 'conv2.2'],
            ['res1.0.0', 'res1.0.1', 'res1.0.2'],
            ['res1.1.0', 'res1.1.1', 'res1.1.2'],
            ['conv3.0', 'conv3.1', 'conv3.2'],
            ['conv4.0', 'conv4.1', 'conv4.2'],
            ['res2.0.0', 'res2.0.1', 'res2.0.2'],
            ['res2.1.0', 'res2.1.1', 'res2.1.2']
        ]
        fused_model = torch.quantization.fuse_modules(self._model, fuse_modules_list)
        fused_model.eval()

        quantization_config = torch.quantization.get_default_qconfig(self.BACKEND)
        fused_model.qconfig = quantization_config
        torch.quantization.prepare(fused_model, inplace=True)

        test_imgs = ImageFolder(calib_data_path, transform=transforms.ToTensor())
        calibration_data_loader = DataLoader(test_imgs, 64, shuffle=True, num_workers=4, pin_memory=True)
        print('CALIBRATING THE MODEL USING THE DATA')
        for batch in tqdm(calibration_data_loader):
            inputs, _ = batch
            fused_model(inputs)

        print('QUANTIZING THE MODEL')
        quantized_model = torch.quantization.convert(fused_model, inplace=True)

        torch.save(quantized_model.state_dict(), self._quant_model_path)
        print(f'QUANTIZATION DONE, SAVED THE MODEL TO: {self._quant_model_path}')
