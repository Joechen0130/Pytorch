import torch
from PIL import Image

yolo5 = torch.hub.load('ultralytics/yolov5', 'yolov5s')
image = Image.open("test.jpg")
image.show()

result = yolo5(image)
result.show()
result.save("result")