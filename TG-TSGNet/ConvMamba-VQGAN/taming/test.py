import os
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from torchvision.transforms import ToPILImage


input_dir = "/data1/yifan/data/vqganfid/valinput12235/"
output_dir = "1" 


os.makedirs(output_dir, exist_ok=True)


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.permute(1, 2, 0))
])

def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    return getattr(__import__(module, fromlist=[cls]), cls)(**config.get("params", {}))


model_config = {
    "target": "taming.models.vqgan_tcm18.VQModel",
    "ckpt_path": "/data1/yifan/taming-transformers-master/logs/2024-11-20T16-50-18_vqgan_tcm18/checkpoints/last.ckpt",
    "params": {
        "embed_dim": 256,
        "n_embed": 1024,
        "ckpt_path": "/data1/yifan/taming-transformers-master/logs/2024-11-20T16-50-18_vqgan_tcm18/checkpoints/last.ckpt",
        "ddconfig": {
            "double_z": False,
            "z_channels": 256,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 64,
            "ch_mult": [1, 1, 2, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [16],
            "dropout": 0.0
        },
        "lossconfig": {
            "target": "taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
            "params": {
                "disc_conditional": False,
                "disc_in_channels": 3,
                "disc_start": 10000,
                "disc_weight": 0.8,
                "codebook_weight": 1.0
            }
        }
    }
}

model = instantiate_from_config(model_config)
checkpoint_path = "/data1/yifan/taming-transformers-master/logs/2024-11-20T16-50-18_vqgan_tcm18/checkpoints/last.ckpt"
state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
model.load_state_dict(state_dict, strict=False)
model.eval().cuda()  


for image_name in tqdm(os.listdir(input_dir), desc="Processing Images"):
    input_path = os.path.join(input_dir, image_name)
    output_path = os.path.join(output_dir, image_name)

    try:
       
        image = Image.open(input_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).cuda()
        print(f"Processing {image_name}: input_tensor shape = {input_tensor.shape}")  
        
        batch = {model.image_key: input_tensor}

        with torch.no_grad():
            log = model.log_images(batch)

        reconstructed_image = log["reconstructions"].squeeze(0).cpu()
        reconstructed_image = torch.clamp(reconstructed_image, -1., 1.)
        grid = reconstructed_image
        grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
        grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
        grid = grid.numpy()
        grid = (grid*255).astype(np.uint8)
        
        Image.fromarray(grid).save(output_path)
        
        


    except Exception as e:
        print(f"Error processing {image_name}: {e}")
