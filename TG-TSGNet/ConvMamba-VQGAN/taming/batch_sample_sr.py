import argparse
import os
from PIL import Image
import torch
from tqdm import tqdm
from torchvision import transforms
from models.sr import SRNO, SharedLayers
import models
from utils import make_coord


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell)
            ql = qr
            preds.append(pred)
        pred = torch.cat(preds, dim=2)
    return pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help="Directory of input images")
    parser.add_argument('--output_dir', required=True, help="Directory to save the output images")
    parser.add_argument('--model', default="/data1/yifan/taming-transformers-master/logs/2024-11-26T10-29-14_c2fmn_13/checkpoints/sr_epoch_end_model.ckpt")
    parser.add_argument('--target_h', type=int, default=256, help="Target height of output image")
    parser.add_argument('--target_w', type=int, default=256, help="Target width of output image")
    parser.add_argument('--gpu', default='0', help="GPU ID")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    scale_max = 4

    # Load model and setup
    shared_layers = SharedLayers().cuda()
    model = SRNO(shared_layers, width=256, blocks=16)

    checkpoint = torch.load(args.model, map_location='cuda')
    state_dict = {key.replace('model.', ''): value for key, value in checkpoint.items()}
    model.load_state_dict(state_dict, strict=True)
    model = model.cuda().eval()

    # Prepare the target size
    h = args.target_h
    w = args.target_w

    # Image transformations
    transform_to_tensor = transforms.ToTensor()
    transform_to_pil = transforms.ToPILImage()

 
    os.makedirs(args.output_dir, exist_ok=True)


    image_list = [img for img in os.listdir(args.input_dir) if img.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]

    for image_name in tqdm(image_list, desc="Processing images"):
    
        image_path = os.path.join(args.input_dir, image_name)

        # Load the image
        img = transform_to_tensor(Image.open(image_path).convert('RGB'))

        # Calculate scale factors
        scale_h = round(h / img.shape[-2], 1)
        scale_w = round(w / img.shape[-1], 1)

        if scale_h > scale_max or scale_w > scale_max:
            print(f"Warning: scale_h ({scale_h}) or scale_w ({scale_w}) exceeds the maximum allowed value of {scale_max}.")
        else:
            # Proceed with the image scaling
            coord = make_coord((h, w), flatten=False).cuda()
            cell = torch.ones(1, 2).cuda()
            cell[:, 0] *= 2 / h
            cell[:, 1] *= 2 / w

            cell_factor_h = max(scale_h / scale_max, 1)
            cell_factor_w = max(scale_w / scale_max, 1)
            cell *= torch.tensor([cell_factor_h, cell_factor_w]).cuda()

            # Model inference
            pred = model(((img - 0.5) / 0.5).cuda().unsqueeze(0), coord.unsqueeze(0), cell).squeeze(0)
            pred = (pred * 0.5 + 0.5).clamp(0, 1).reshape(3, h, w).cpu()

            # Save the output image
            output_path = os.path.join(args.output_dir, image_name)
            transform_to_pil(pred).save(output_path)
            # print(f"Saved {image_name} to {output_path}")
