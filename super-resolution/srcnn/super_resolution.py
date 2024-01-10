import argparse
import os
import torch
from skimage import io
import torchvision.transforms.functional as F
from model.srcnn import Net
import utils

def super_resolution(input_dir, output_dir, model, params):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            input_path = os.path.join(subdir, file)
            print(f"Processing: {input_path}")

            try:
                image = io.imread(input_path)
                image = F.to_tensor(image).unsqueeze(0)

                # Resize the image to match the expected input size of SRCNN
                image = F.resize(image, (params.image_size, params.image_size))
                
                image = image.cuda() if torch.cuda.is_available() else image

                with torch.no_grad():
                    output = model(image)

                output = output.cpu().squeeze(0)
                output = F.to_pil_image(output)

                output_path = os.path.join(output_dir, file)
                output.save(output_path)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

def main(args, params):
    model = Net(params)
    
    checkpoint = torch.load(args.model_path, map_location='cpu')  # Load the model
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()

    super_resolution(args.input_dir, args.output_dir, model, params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help="Directory containing low-resolution images")
    parser.add_argument('--output_dir', required=True, help="Directory to save the high-resolution images")
    parser.add_argument('--model_path', required=True, help="Path to the trained SRCNN model")

    args = parser.parse_args()

    # Load parameters from the model directory
    model_params = utils.Params(os.path.join(os.path.dirname(args.model_path), 'params.json'))

    main(args, model_params)
