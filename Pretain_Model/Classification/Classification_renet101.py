from torchvision import models,transforms
import torch
from PIL import Image

resnet = models.resnet101(pretrained=True)

#print(resnet)
#把resnet當成函式一般呼叫，輸入為圖片，輸出為1000個ImageNet類別標籤的信心分數。

#對圖片進行預先處理:
#利用transforms 函式庫
prepocess = transforms.Compose([
    transforms.Resize(256),#轉換尺寸
    transforms.CenterCrop(224),#剪裁
    transforms.ToTensor(),#轉為張量
    transforms.Normalize(
        mean=[0.485,0.456,0.406],#各色彩通道平均值和標準差
        std=[0.229,0.224,0.225]
    )
])

image = Image.open("bobby.jpg")
image_preprocess = prepocess(image)
#print(image_preprocess.shape)#[3,224,224]

#因為resnet輸入的資料必須是4維
#第一維代表batch_size
#利用unsqueeze()

batch_image = torch.unsqueeze(image_preprocess,0)#在第0階處
#print(batch_image.shape)#[1,3,224,224]

#進行推論
#模型設為eval模式
resnet.eval()
result = resnet(batch_image)
#print(result)
#result為2維 :第一個為batch_size,第二個為1000類的信心度

with open("imagenet_classes.txt")as f:
    labels = [line.strip() for line in f.readlines()]

#取得結果張量最大值之索引
_, index = torch.max(result, 1)
#print(index)#tensor([207])
#print(index[0])#tensor(207)
percentage = torch.nn.functional.softmax(result, dim=1)[0] * 100
print(labels[index[0]], percentage[index[0]].item())

_,indices =torch.sort(result, descending = True)
print([(labels[idx],percentage[idx].item()) for idx in indices[0][:5]])