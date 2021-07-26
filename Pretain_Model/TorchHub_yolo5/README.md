# 使用 TorchHub 來載入別人 github 的預訓練模型
###在 google 中找尋有 _**hubconf.py**_ 的github專案  
###我使用使用 [ultralytics/yolov5](https://github.com/ultralytics/yolov5/blob/master/hubconf.py) 的預訓練模型測試  
## TorchHub 載入預訓練模型  

    yolo5 = torch.hub.load('ultralytics/yolov5', 'yolov5s')
##  載入原圖  

    image = Image.open("test.jpg")
    image.show()
![]()

## 偵測結果並檢視儲存  
    result = yolo5(image)
    result.show()
    result.save("result")
![]()