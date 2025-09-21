# UMKD
Code for the paper "[Uncertainty-Aware Multi-Expert Knowledge Distillation for Imbalanced Disease Grading](https://arxiv.org/abs/2505.00592)", published in MICCAI-2025.
## Datasets

Dataset   |        URL       
:--------------:|:------------------:|
Diabetic Retinopathy  |   [APTOS_2019](https://www.kaggle.com/datasets/mariaherrerot/aptos2019)              
Diabetic Retinopathy   |  [Eyepacs](https://zhuanlan.zhihu.com/p/683930522)        
Prostate Cancer  |   [SICAPv2](https://zhuanlan.zhihu.com/p/686314573) 

## Train
### Experts
Expert Model   |        Code     |    num_classes   
:--------------:|:------------------:|:--------------------:
ResNet50        |   resnet_linear_dr.py        |     Diabetic Retinopathy(5) / Prostate Cancer(4)             
ResNet50        |   resnet_linear_dr.py        |     Diabetic Retinopathy(5) / Prostate Cancer(4)      

### Student
Target Model    |     Code       |      Methods 
:--------------:|:-----------:|:-------------------:
ResNet18        |   Resnet_trainer_UMKD.py    |      Knowledge Distillation with UMKD
