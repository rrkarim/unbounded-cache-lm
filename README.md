# Unbounded cache model for online language modeling with open vocabulary
(https://arxiv.org/abs/1711.02604)

## Requirement
Chainer v1.15.0.1

### Training
`model.py` contains code for building model.  
You can start training like following command:
```shell
$ python main.py --batchsize=64 --gpu=0 --embed=128 --unit=256 --out=result -L=30 --mode=train
```

## Reference
Edouard Grave, Moustapha Cisse, Armand Joulin
(https://arxiv.org/abs/1711.02604)

## TODO

- [+] IVFPQ (https://arxiv.org/abs/1702.08734)
