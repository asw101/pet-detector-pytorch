
from datetime import datetime
from PIL import Image
from torchvision import transforms
from urllib.request import urlopen

import logging
import os
import sys
import torch

from azure.storage.blob import BlobServiceClient

model_path = 'checkpoint.pth'
if 'connect_str' in os.environ:
    connect_str = os.environ['connect_str']
else:
    raise Exception('msg', 'connection string not found')
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
blob_client = blob_service_client.get_blob_client(container='petdetector', blob='checkpoint.pth')

with open(os.path.join(os.getcwd(), model_path), "wb") as my_blob:
    download_stream = blob_client.download_blob()
    my_blob.write(download_stream.readall())

model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

def get_class_labels():
    classes = {}

    try:
        dirname = os.path.dirname(__file__)
        with open(os.path.join(dirname, 'labels.txt'), 'r') as f:
            classes = f.read().splitlines() 
    except FileNotFoundError:
        logging.info(os.listdir(os.curdir))
        logging.info(os.curdir)
        raise

    return classes

def predict_image_from_url(image_url):
    class_dict = get_class_labels()
    with urlopen(image_url) as testImage:
        try:
            input_image = Image.open(testImage).convert('RGB')
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')

            with torch.no_grad():
                output = model(input_batch)
            # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
            print(output[0])
            # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
            softmax = (torch.nn.functional.softmax(output[0], dim=0))
            out = class_dict[softmax.argmax().item()]

            response = {
                'created': datetime.utcnow().isoformat(),
                'predictedTagName': out,
                'prediction': softmax.max().item()
            }
        except:
            response = {
                'error' : 'image url is wrong'
            }

        logging.info(f'returning {response}')
        return response

if __name__ == '__main__':
    predict_image_from_url(sys.argv[1])