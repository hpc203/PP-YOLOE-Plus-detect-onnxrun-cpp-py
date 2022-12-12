# PP-YOLOE-Plus-detect-onnxrun-cpp-py
使用ONNXRuntime部署百度飞桨开源PP-YOLOE-Plus目标检测，包含C++和Python两个版本的程序.
起初，我想使用opencv做部署的，但是opencv的dnn模块读取onnx文件出错， 
无赖只能使用onnxruntime做部署了。本套程序一共提供了4个onnx模型， onnx文件需要从百度云盘下载，
链接：https://pan.baidu.com/s/1K34jLRORTK8IYr8X0jXH4g 
提取码：ehey

注意onnxruntime的版本，起初我的电脑安装的是onnxruntime1.8，但是在读取onnx的文件的时候，出错了。
升级onnxruntime到1.11.1，就能正常推理了
