import argparse
import random
import os
from tqdm import tqdm
from skimage import io
from skimage.transform import resize, downscale_local_mean
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
import imageio
from skimage import color

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/High', help="Directory with the dataset")
parser.add_argument('--output_dir', default='../data/CNN_Output', help="Where to write the new data")
parser.add_argument('--input_size', default='144', help="Size of the input images")
parser.add_argument('--output_size', default='144', help="Size of the output images")
parser.add_argument('--up_scale', default='4', help="Upscaling factor")

def crop_and_save(filename, output_dir, out_size):
    # Crop the image contained in `filename` and save it to the `output_dir`
    image = io.imread(filename)
    # Check if the image has an alpha channel (transparency)
    if image.ndim == 3 and image.shape[2] == 4:
        # Convert RGBA to RGB
        image = color.rgba2rgb(image)   
    cropped = resize(image, (out_size, out_size), preserve_range=True).astype(np.uint8)
    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0] + '.jpg')
    imageio.imsave(output_path, cropped, format='jpg')  # Save in JPG format

def blur_and_save(filename, output_dir, in_size, out_size, up_scale):
    # Blur the image contained in `filename` and save it to the `output_dir`
    image = io.imread(filename)
    cropped = resize(image, (out_size, out_size), preserve_range=True).astype(np.uint8)
    downscaled = resize(cropped, (out_size // up_scale, out_size // up_scale), preserve_range=True).astype(np.uint8)
    blur = resize(downscaled, (out_size, out_size), preserve_range=True).astype(np.uint8)

    # Check if the image has an alpha channel (transparency)
    if blur.ndim == 3 and blur.shape[2] == 4:
        # Convert RGBA to RGB
        blur = color.rgba2rgb(blur)

    # Ensure the data type is compatible with Pillow
    blur = blur.astype(np.uint8)

    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0] + '.jpg')
    imageio.imsave(output_path, blur, format='jpg')  # Save in JPG format

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Could not find the dataset at {}".format(args.data_dir)

    # get args
    data_dir = args.data_dir
    INPUT_SIZE = int(args.input_size)
    OUTPUT_SIZE = int(args.output_size)
    UP_SCALE = int(args.up_scale)

    # Get the filenames in data directory
    filenames = os.listdir(data_dir)
    filenames = [os.path.join(data_dir, f) for f in filenames if f.endswith(('.jpg', '.png'))]

    # Split the images into 98% train, 1% val, and 1% test
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split1 = int(0.98 * len(filenames))
    split2 = (len(filenames) - split1) // 2 + split1
    train_filenames = filenames[:split1]
    val_filenames = filenames[split1:split2]
    test_filenames = filenames[split2:]

    print("train", len(train_filenames))
    print("val", len(val_filenames))
    print("test", len(test_filenames))

    filenames = {'train': train_filenames,
                 'val': val_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, val, and test
    for split in ['train', 'val', 'test']:
        output_dir_split_clear = os.path.join(args.output_dir, '{}_clear'.format(split))
        output_dir_split_blur = os.path.join(args.output_dir, '{}_blur'.format(split))

        if not os.path.exists(output_dir_split_clear):
            os.mkdir(output_dir_split_clear)
        else:
            print("Warning: dir {} already exists".format(output_dir_split_clear))

        if not os.path.exists(output_dir_split_blur):
            os.mkdir(output_dir_split_blur)
        else:
            print("Warning: dir {} already exists".format(output_dir_split_blur))

        print("Processing {} data, saving to {} and {}".format(split, output_dir_split_clear, output_dir_split_blur))

        for filename in tqdm(filenames[split]):
            print("Processing file:", filename)
            crop_and_save(filename, output_dir_split_clear, OUTPUT_SIZE)

        for filename in tqdm(filenames[split]):
            print("Processing file:", filename)
            blur_and_save(filename, output_dir_split_blur, INPUT_SIZE, OUTPUT_SIZE, UP_SCALE)

    print("Done building dataset")