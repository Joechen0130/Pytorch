#使用Resnet101預訓練模型來進行圖片辨識
分為三個部分
1.圖片前處理
2.帶入預訓練模型
3.辦識結果

#Step1 : 圖片前處理
使用 transforms 將圖片轉為 Tensor 形式  
    
    prepocess = transforms.Compose([
        transforms.Resize(256),#轉換尺寸
        transforms.CenterCrop(224),#剪裁
        transforms.ToTensor(),#轉為張量
        transforms.Normalize(
            mean=[0.485,0.456,0.406],#各色彩通道平均值和標準差
            std=[0.229,0.224,0.225]
    )])
但由於resnet輸入的資料必須是4維，必須使用 torch.unsqueeze()  
來將3維擴展到4維    
    
    第一維代表batch_size
    batch_image = torch.unsqueeze(image_preprocess,0)#在第0階處

#Step2 : 帶入預訓練模型
模型設為eval模式  
並帶入圖片
    
    resnet.eval()
    result = resnet(batch_image)

#Step3 : 辦識結果
result為2維 :第一個為batch_size,第二個為1000類的信心度  

#Result:  
By classification cat.jpeg
    tabby, tabby cat 74.90569305419922
