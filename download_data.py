from datasets import CUB200, StanfordDogs

CUB200(root='/data3/tongshuo/Grading/CommonFeatureLearning/data/cub200', split='train',
                    download=True)
StanfordDogs(root='/data3/tongshuo/Grading/CommonFeatureLearning/data/dogs', split='train',
                    download=True)