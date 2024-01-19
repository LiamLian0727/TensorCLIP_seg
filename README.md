The repository contains a PyTorch implementation that applies TensorCLIP to semantic segmentation. For the time being, we'll use the open-source CLIP instead of TensorCLIP, and the decoder part will be referenced from DenseCLIP with slight modifications.

### Requirements

- torch>=1.8.0
- torchvision
- timm
- mmcv-full==1.3.17
- mmseg==0.19.0
- mmdet==2.17.0
- regex
- ftfy
- fvcore

To use our code, please first install the `mmcv-full` and `mmseg`/`mmdet` following the official guidelines ([`mmseg`](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md), [`mmdet`](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md)) and prepare the datasets accordingly.

### Pre-trained CLIP Models

Download the pre-trained CLIP models (`RN50.pt`, `RN101.pt`, `VIT-B-16.pt`) and save them to the `pretrained` folder. The download links can be found in [the official CLIP repo](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py#L30).

#### Training & Evaluation on ADE20K

To train this model based on CLIP ViT-B 16 with 2 gpu, run:

```
bash dist_train.sh tensorclip_fpn_vit-b_640x640_80k.py 2
```

To evaluate the performance with multi-scale testing, run:

```
bash dist_test.sh configs/tensorclip_fpn_vit-b_640x640_80k.py work_dirs/tensorclip_fpn_vit-b_640x640_80k/latest.pth 2 --eval mIoU --aug-test
```

To better measure the complexity of the models, we provide a tool based on `fvcore` to accurately compute the FLOPs of `torch.einsum` and other operations:
```
python get_flops.py configs/tensorclip_fpn_vit-b_640x640_80k.py --fvcore
```
You can also remove the `--fvcore` flag to obtain the FLOPs measured by `mmcv` for comparisons.
