# 使用 TorchHub 來載入別人 github 的預訓練模型
###_在 google 中找尋有 _**hubconf.py**_ 的github專案_  
###_我使用 [ultralytics/yolov5](https://github.com/ultralytics/yolov5/blob/master/hubconf.py) 的預訓練模型測試_  
## TorchHub 載入預訓練模型  

    yolo5 = torch.hub.load('ultralytics/yolov5', 'yolov5s')
##  載入原圖  
[圖片來源](https://www.pexels.com/zh-tw/photo/297927/)  

    image = Image.open("test.jpg")
    image.show()
![](https://github.com/Joechen0130/Pytorch/blob/master/Pretain_Model/TorchHub_yolo5/test.jpg)

## 偵測結果並檢視儲存  
    result = yolo5(image)
    result.show()
    result.save("result")
![](https://github.com/Joechen0130/Pytorch/blob/master/Pretain_Model/TorchHub_yolo5/result/test.jpg)