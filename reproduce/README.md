## Reproduce results in our paper

### 5-fold Cross-Validation

#### REALSumm
* Split by examples
```
# Lite2Pyramid
python cross_validation.py --data realsumm --split examples --version 2 --device 0

# Lite2.5Pyramid
python cross_validation.py --data realsumm --split examples --version 2.5 --device 0

# Lite3Pyramid
python cross_validation.py --data realsumm --split examples --version 3 --device 0
```

* Split by systems
```
# Lite2Pyramid
python cross_validation.py --data realsumm --split systems --version 2 --device 0

# Lite2.5Pyramid
python cross_validation.py --data realsumm --split systems --version 2.5 --device 0

# Lite3Pyramid
python cross_validation.py --data realsumm --split systems --version 3 --device 0
```

### Out-of-the-box Generalization

