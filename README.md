## Combined Depth Space based Architecture Search for Person Re-identification
### Paper
[Combined Depth Space based Architecture Search for Person Re-identification](https://arxiv.org/abs/2104.04163)
### Models

- results on ReID tasks

| model             | Market(mAP/rank-1) | Duke(mAP/rank-1) | MSMT17(mAP/rank-1) |
| ----------------- | :----------------: | :--------------: | :----------------: |
| cnet(scratch)     |     83.5/93.6      |    73.2/86.0     |     47.7/73.3      |
| cdnet(scratch)    |     83.7/93.7      |    73.9/86.7     |     48.5/73.7      |
| cdnet(pretrained) |     86.0/95.1      |    76.8/88.6     |     54.7/78.9      |

- results on classification 

| model          | Cifar-100(acc/param) | ImageNet(acc/param) |
| -------------- | -------------------- | ------------------- |
| cdnet(scratch) | 82.1/2.3M            | 75.1/2.5M           |

### Evaluation

You can download the models from [here](https://github.com/solicucu/models) firstly and then run the script in "./run/" according the need.

For example,  test the cdnet(pretrained) on Market1501, modify the "./configs/inferences.yml" with according values as follows:

```yaml
MODEL:
    NAME: 'cdnet'
    GENOTYPE: "cdnet_sample_top2_best_genotype.json"
DATA:
    DATASET: 'market1501'
    DATASET_DIR: "/home/share/solicucu/data/" # path to the dataset DATASET
    IMAGE_SIZE: [256,128]
OUTPUT:
    DIRS: "/home/share/solicucu/data/ReID/FasterReID/inference/"
    CKPT_DIRS: "market1501/" # DIRS + CKPT_DIRS is path to the checkpoint 
TEST:
    BEST_CKPT: "cdnet_top2_pretrained.pth" # name of the specified checkpoint
```

Then run the file "./run/inference.sh".

Note that this file can be used for the evaluation of both ReID task and classification. As for evaluation on Cifar100 and ImageNet,  run the file "./run/infer_cifar.sh" and "./infer_imagenet.sh" respectively. Specially, it need to change a according FBLNeck for ImageNet by using it in "./model/head/imagenet_bl_neck.py", where the FBLNeck is simplified without fine-grained part.  Therefore, you can change it in "./model/heads/__ init __ .py" easily.

### Training

- top-k sample search 

  As for cdnet,

  1.modify the configuration in "./factory/cdnet_sample_topk_search/config.py" if necessary.

  2.execute the command "python train_search.py"

  The directory "cnet_sample_topk_search" is used for cnet accordingly.

- train from scratch (all network)

  1.modify the configuration files 

  2.run the according scripts in "./run"

  For example,  train the cdnet on market1501, modify the file "./configs/cdnet.yml",

  ```yaml
  DATA:
    DATASET: 'market1501'
  OUTPUT:
    DIRS: "/home/share/solicucu/data/ReID/FasterReID/market1501/cdnet/"
    CKPT_DIRS: "checkpoints/cdnet_top2_fblneck/"
    LOG_NAME: 'log_cdnet_top2_fblneck.txt'
  ```

  Note that DIRS + CKPT_DIRS is path to save the checkpoint .

  

- train from pretrained models(cnet/cdnet)

  1.train the cdnet or cnet on Imagenet and  obtain the pretrained checkpoint.

  2.modify the configuration files "./configs/anynet_pretrained.yml"

  3.run the script "./run/anynet_pretraiend.sh"

  For example, train the cdnet on market1501, modify the file "./configs/anynet_pretrained.yml"

  ```yaml
  MODEL:
    NAME: 'cdnet'
    IMAGENET_CKPT: 'path/to/pretained_chekcpoint'
    GENOTYPE: "cdnet_sample_top2_best_genotype.json"
  DATA:
    DATASET: 'market1501'
  SOLVER:
    MAX_EPOCHS: 350
    BASE_LR: 6.5e-2
    LR_LIST: [6.5e-2, 6.5e-3, 6.5e-4, 6.5e-5]
  OUTPUT:
    DIRS: "/home/share/solicucu/data/ReID/FasterReID/market1501/cdnet/"
    CKPT_DIRS: "checkpoints/cdnet_top2_fblneck_pretrained/"
    LOG_NAME: 'log_cdnet_top2_fblneck_pretrained.txt'
  ```

  Specially, there are a little adjustment for msmt17 as follows.

  ```yaml
  SOLVER:
    MAX_EPOCHS: 240
    BASE_LR: 4.5e-2
    LR_LIST: [4.5e-2, 4.5e-3, 4.5e-4, 4.5e-5]
  ```

  Other modifications can be made according the need. 



