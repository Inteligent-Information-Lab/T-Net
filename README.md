# T-Net #
----------
This is the PyTorch implementation for our paper:

**T-Net: Deep Stacked Scale-Iteration Network for Image Dehazing**

### Environment ###
----------
- python >= 3.6
- pytorch >= 1.0
- torchvision >= 0.2.2
- pillow >= 5.1.0
- numpy >=1.14.3
- scipy >= 1.1.0
- skitch-image >= 0.15.0

### Training ###
----------
- Dataset Google drive: https://drive.google.com/file/d/19pwuhQd1yBgA939AjbXTdkoZ91yiD4Kw/view?usp=sharing
- Tarin Stack T-Net:
```bash
$ python trian.py -recurrent_iter 3 -updown_pairs 4 -rdb_pairs 3
```

### Test on synthetic data ###
----------
- Dataset Google drive: https://drive.google.com/file/d/17KaYJXKkLU-5my7HUvEoTSREzzoyTAP7/view?usp=sharing
- Test Stack T-Net:
```bash
$ python test.py -recurrent_iter 3 -updown_pairs 4 -rdb_pairs 3
```

### Test on real-world data ###
----------
- Dataset Google drive: https://drive.google.com/file/d/1j7p8B8WdmrF3G4u8iwJXCbZrFsM7faHu/view?usp=sharing
- Test Stack T-Net:
```bash
$ python apply.py -recurrent_iter 3 -updown_pairs 4 -rdb_pairs 3
```
