from PIL import Image
import paddle
import paddle.vision.transforms as transforms


def load_image(image_path):
    return Image.open(image_path).convert('RGB')


def apply_transforms(image: Image.Image, size=224) -> paddle.Tensor:
    if not isinstance(image, Image.Image):
        raise Exception("Type error! at func apply_transforms")
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    tensor = transform(image).unsqueeze(0)

    tensor.stop_gradient = False

    return tensor


def denormalize(tensor: paddle.Tensor):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    # bug 传进来的是paddle Tensor
    denormalized = tensor.clone()

    for channel, mean, std in zip(denormalized[0], means, stds):
        channel.multiply(paddle.to_tensor(std, 'float32')). \
            add(paddle.to_tensor(mean, 'float32'))

    return denormalized


def standardize_and_clip(tensor: paddle.Tensor, min_value=0.0, max_value=1.0,
                         saturation=0.1, brightness=0.5):
    tensor = tensor.detach().cpu()

    mean = tensor.mean()
    std = tensor.std()

    if std == 0:
        std += 1e-7

    standardized = tensor.subtract(mean).divide(std).multiply(paddle.to_tensor(saturation, 'float32'))
    clipped = standardized.add(paddle.to_tensor(brightness)).clip(paddle.to_tensor(min_value), paddle.to_tensor(max_value))

    return clipped


def format_for_plotting(tensor):
    """Formats the shape of tensor for plotting.
    """

    has_batch_dimension = len(tensor.shape) == 4
    has_no_channel_dimension = len(tensor.shape) == 2
    formatted = tensor.clone()

    if has_batch_dimension:
        formatted = tensor.squeeze(0)

    if has_no_channel_dimension:
        formatted = tensor.unsqueeze(0)

    if formatted.shape[0] == 1:
        return formatted.squeeze(0).detach()
    else:
        return formatted.transpose(perm=[1, 2, 0]).detach()
