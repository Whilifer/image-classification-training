from PIL import Image
from torchvision import transforms

TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ]
)


def preprocess(image: Image.Image):
    tensor = TRANSFORM(image)
    return tensor.unsqueeze(0)
