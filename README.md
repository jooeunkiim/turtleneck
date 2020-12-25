# 노트북 자세 판별기
2020-2 YBIGTA 컨퍼런스


### References

* [Facial landmarks with dlib, OpenCV, and Python](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)
* [Detect eyes, nose, lips, and jaw with dlib, OpenCV, and Python](https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/)
* [Yaw Pitch Roll Detection using Retina Face](https://github.com/fisakhan/Face_Pose)
* [Simple Random Forest Classification Example](https://github.com/codebasics/py/blob/master/ML/11_random_forest/11_random_forest.ipynb)
* [Random Forest Hyperparameter Tuning](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)
* [Saving Models with Scikit-learn](https://scikit-learn.org/stable/modules/model_persistence.html)

### Description

Pitch, Roll, Yaw는 회전 축을 세 개로 쪼개놓은 것이다. 

![Orientation of the head in terms of pitch , roll , and yaw movements... |  Download Scientific Diagram](https://www.researchgate.net/profile/Tsang_Ing_Ren/publication/279291928/figure/fig1/AS:292533185462272@1446756754388/Orientation-of-the-head-in-terms-of-pitch-roll-and-yaw-movements-describing-the-three.png)

얼굴의 크기(Width, Height)와 Pitch가 자세와 유의미한 상관관계가 있음을 보여준다.

![image-20201225200338221](./imgs/image-20201225200338221.png)

### CNN에서 보완할 점들

* 옷과 배경에 따라서 결과값이 변화
* 노트북 화각에 따라서 결과값이 변화
* 거리에 따라서 성능 변화

즉, 사진 안에 있는 요소들이 성능에 방해하는 경우가 있었음.

### 해결방법

얼굴 외적인 요소들을 통제하기 위해서 face coordinates을 그려넣고, 이를 이용함.

이 때 선택의 기로가 있었음. Dlib vs retinaface(tf2)

Dlib: 300Mb

Retinaface(tf2): 2Mb

