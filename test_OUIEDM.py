import os
import sys

sys.path.append(os.getcwd())
import glob
import argparse
import torch
from torchvision import transforms
from PIL import Image
from ouiedm import test_inference

from ram.models.ram_lora import ram
from ram import inference_ram as inference

tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512))
])

ram_transforms = transforms.Compose([

    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_validation_prompt(args, image, model):
    lq = image
    lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
    captions = inference(lq_ram, model)
    validation_prompt = f"{captions}, {args.prompt},"

    return validation_prompt, lq

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str,
                        default='your input_path',
                        help='path to the input image')
    parser.add_argument('--output_dir', '-o', type=str,
                        default='your output_path',
                        help='the directory to save the output')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='stabilityai/stable-diffusion-2-1-base',
                        help='sd` model path')
    parser.add_argument('--seed', type=int, default=35, help='Random seed to be used')
    parser.add_argument("--checkpoint_path", type=str, default='your checpoint_path')
    parser.add_argument('--prompt', type=str, default='', help='user prompts')
    parser.add_argument('--ram_path', type=str,
                        default='https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/raw_swin_large_14m.pth')

    parser.add_argument('--save_prompts', type=bool, default=True)
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16")
    parser.add_argument("--merge_and_unload_lora", default=False)

    args = parser.parse_args()

    model = test_inference(args)
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.*'))
    else:
        image_names = [args.input_image]

    # get ram model
    RAM = ram(pretrained=args.ram_path,
               pretrained_condition=None,
               image_size=384,
               vit='swin_l')
    RAM.eval()
    RAM.to("cuda")

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    RAM = RAM.to(dtype=weight_dtype)

    if args.save_prompts:
        txt_path = os.path.join(args.output_dir, 'txt')
        os.makedirs(txt_path, exist_ok=True)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')
    for image_name in image_names:

        input_image = Image.open(image_name).convert('RGB')
        bname = os.path.basename(image_name)
        input_image = tensor_transforms(input_image).unsqueeze(0).to("cuda")
        # get caption
        validation_prompt, lq = get_validation_prompt(args, input_image, RAM)
        if args.save_prompts:
            txt_save_path = f"{txt_path}/{bname.split('.')[0]}.txt"
            with open(txt_save_path, 'w', encoding='utf-8') as f:
                f.write(validation_prompt)
                f.close()
        print(f"process {image_name}, tag: {validation_prompt}".encode('utf-8'))

        with torch.no_grad():
            lq = lq * 2 - 1
            output_image = model(lq, prompt=validation_prompt)
            output_put = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        output_put.save(os.path.join(args.output_dir, bname))

