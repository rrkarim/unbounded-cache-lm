# Unbounded cache model for online language modeling with open vocabulary
(https://arxiv.org/abs/1711.02604)

## Requirement
faiss

### Training
```
TRAIN_PATH=data/toy_reverse/train/data.txt
DEV_PATH=data/toy_reverse/dev/data.txt
$ python main.py --train_path $TRAIN_PATH --dev_path $DEV_PATH
```

## Reference
Edouard Grave, Moustapha Cisse, Armand Joulin
(https://arxiv.org/abs/1711.02604)

## TODO

- [+] IVFPQ (https://arxiv.org/abs/1702.08734)
