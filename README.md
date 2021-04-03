# Project of “Feature Selection Module for CNN Based Object Detector”

## note
This project support YoloV3, YoloV4 and YoloV5.

## install 
```pip install -r requirements.txt -i https://pypi.douban.com/simple```

## Environment

* Nvida GeForce TitanX
* CUDA10.0
* CUDNN7.0
* ubuntu 18.04
* python 3.6

## pretrained weights
Baidu Netdisk: https://pan.baidu.com/s/1kxuHjR7qZTwGN3Z1y02AFw  9xja

## data prepare
```
|${folder}
 |----YOLO-FSM
 |----weights
 |----dataset
      |----VOCtest-2007
           |----VOCdevkit
           |    |----VOC2007
           |         |----Annotations
           |         |----JPEGImages
           |         |----ImageSets
           |    
      |----VOCtrainval-2007
           |----VOCdevkit
           |    |----VOC2007
           |         |----Annotations
           |         |----JPEGImages
           |         |----ImageSets
           | 
      |----VOCtrainval-2012
           |----VOCdevkit
           |    |----VOC2012
           |         |----Annotations
           |         |----JPEGImages
           |         |----ImageSets
           | 
```

## Train or Test
please set yolo model in train.py or test.py
```
cd utils; python voc.py; cd -
python train.py --weight_path ${weight_path}
python test.py --weight_path ${weight_path}
```

## Result on Pascal VOC2007-test
| Name | mAP | aero | bike | bird | boat | bottle | bus | car | cat | chair | cow | table | dog | horse | motor | person | plant | sheep | sofa | train | monitor | 
| :----- | :----- | :------ | :----- | :-----| :-----| :-----| :-----| :-----| :-----| :-----| :-----| :-----| :-----| :-----| :-----| :-----| :-----| :-----| :-----| :-----| :-----| 
| FasterRCNN  | 73.2 | 76.5 | 79.0 | 70.9 | 65.5 | 52.1 | 83.1 | 84.7 | 86.4 | 52.0 | 81.9 | 65.7 | 84.8 | 84.6 | 77.5 | 76.7 | 38.8 | 73.6 | 73.9 | 83.0 | 72.6 |
| SSD300	  | 74.3 | 75.5 | 80.2 | 72.3 | 66.3 | 47.6 | 83.0 | 84.2 | 86.1 | 54.7 | 78.3 | 73.9 | 84.5 | 85.3 | 82.6 | 76.2 | 48.6 | 73.9 | 76.0 | 83.4 | 74.0 |
| HyperNet	  | 76.3 | 77.4 | 83.3 | 75.0 | 69.1 | 62.4 | 83.1 | 87.4 | 87.4 | 57.1 | 79.8 | 71.4 | 85.1 | 85.1 | 80.0 | 79.1 | 51.2 | 79.1 | 75.7 | 80.9 | 76.5 |
| CenterNet	  | 81.5 | 89.6 | 89.0 | 79.3 | 73.3 | 75.3 | 86.6 | 89.6 | 86.6 | 67.0 | 87.1 | 75.1 | 85.6 | 90.0 | 87.0 | 86.2 | 58.7 | 80.5 | 73.9 | 87.9 | 81.6 |
| YoloV3	  | 81.0 | 88.0 | 87.4 | 81.8 | 69.6 | 73.7 | 85.7 | 88.6 | 87.0 | 67.2 | 85.7 | 74.6 | 87.1 | 87.7 | 84.9 | 85.6 | 58.9 | 84.1 | 76.1 | 84.6 | 81.1 |
| YoloV3-SE	  | 81.2 | 89.5 | 86.7 | 80.8 | 73.7 | 69.9 | 86.8 | 89.0 | 88.1 | 66.5 | 85.8 | 74.2 | 86.6 | 89.2 | 86.2 | 85.1 | 57.9 | 86.5 | 75.1 | 87.5 | 79.1 |
| YoloV3-FSM  | 81.6 | 87.3 | 88.2 | 82.2 | 73.6 | 69.9 | 85.9 | 88.9 | 87.2 | 68.6 | 87.6 | 76.7 | 86.8 | 89.8 | 86.0 | 85.7 | 57.3 | 86.4 | 77.9 | 85.8 | 81.1 |
| YoloV4	  | 83.9 | 90.1 | 89.1 | 83.3 | 76.7 | 76.4 | 90.2 | 89.5 | 88.6 | 73.0 | 89.4 | 80.4 | 87.6 | 90.1 | 88.4 | 87.9 | 57.7 | 85.9 | 82.2 | 87.6 | 83.8 |
| YoloV4-FSM  | 85.0 | 90.1 | 89.9 | 84.7 | 78.9 | 78.3 | 90.0 | 89.9 | 88.9 | 74.7 | 89.3 | 82.8 | 87.8 | 90.4 | 89.5 | 88.4 | 62.4 | 87.2 | 82.0 | 88.5 | 87.0 |
| YoloV5l	  | 80.7 | 89.6 | 87.2 | 78.3 | 71.5 | 74.9 | 86.1 | 89.4 | 86.1 | 64.8 | 86.6 | 75.5 | 84.3 | 90.0 | 85.6 | 85.6 | 56.3 | 83.5 | 75.5 | 85.2 | 78.2 |
| YoloV5l-FSM | 82.2 | 86.6 | 88.6 | 82.4 | 76.5 | 75.6 | 87.0 | 89.4 | 87.9 | 66.1 | 87.2 | 78.0 | 85.9 | 88.1 | 88.3 | 86.6 | 58.6 | 87.4 | 74.7 | 87.7 | 80.4 |

## Visualization on VOC2007 Test set
Baidu Netdisk: https://pan.baidu.com/s/1FySdBBbZzkjyKr0KN9JpSQ  hn8t
