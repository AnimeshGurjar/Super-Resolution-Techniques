from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import io

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        img_summaries = []

        for i, img in enumerate(images):
            # Convert the image array to a PIL Image
            img = Image.fromarray((img * 255).astype(np.uint8))

            # Save the image to a BytesIO buffer
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")

            # Create a tensor from the buffer and add it to the summary
            img_tensor = np.array(Image.open(buffer))
            img_summaries.append(self.create_image_summary(tag + '/{}'.format(i), img_tensor))

        # Add image summaries to the writer
        self.writer.add_images(tag, img_summaries, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(tag, values, step, bins=bins)

    def create_image_summary(self, tag, img):
        """Create an image summary."""
        return self.writer.image(tag, img, dataformats="HWC")

    def close(self):
        """Close the writer."""
        self.writer.close()