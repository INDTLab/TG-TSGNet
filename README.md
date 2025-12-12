This repository provides code and datasets related to the paper: 

### **TG-TSGNet: A Text-Guided Arbitrary-Resolution Terrain Scene Generation Network**.



### Dataset Download

[Natural Terrain Scene Data Set (NTSD)](https://drive.google.com/drive/folders/1fEGIvaNngXxSVOGn6fWOLiAo1FC9ApVR?usp=drive_link) from **Terrain Scene Generation Using A Lightweight Vector Quantized Generative Adversarial Network**

[Natural Terrain Scene Textual Description Data Set (NTSD-TD)](https://drive.google.com/drive/folders/1omyvk49UFd25DJZaaQwLJO5CKMj_np-9?usp=sharing)



### Installation

Create and activate a new conda environment:

```
conda create -n TG-TSGNet python=3.9.0 -y
conda activate TG-TSGNet
pip install -r requirements.txt
```



### Training

To train the ConvMamba-VQGAN model, run:

```
python ConvMamba-VQGAN/taming/train_vqsr.py --base configs/vqgan_sr.yaml --gpus 7, -t
```

Training the Encoder Only,run:

```
python ConvMamba-VQGAN/taming/main.py --base configs/vqgan_mgb.yaml --gpus 7, -t
```

Training the ARSRM Only,run:

```
ConvMamba-VQGAN/SR/train.py --config configs/train_sample.yaml --gpu 7 --name 
```



Training Text Guidance Sub-network,run:

```
train.py --image_text_folder xxx --taming --batch_size 32 --vqgan_model_path xxx.pt --vqgan_config_path xxx.yaml --output_file_name xxx
```



### Citation

If you find our work useful in your research, please consider citing:
```
@ARTICLE{TIP_35094_2025,
    author={Yifan Zhu, Yan Wang, Xinghui Dong},
    journal={IEEE Transactions on Image Processing}, 
    title={TG-TSGNet: A Text-Guided Arbitrary-Resolution Terrain Scene Generation Network}, 
    year={2025},
    }
```


