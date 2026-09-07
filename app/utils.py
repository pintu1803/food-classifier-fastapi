from torchvision import transforms
from config import IMAGE

def myLog(title):
    margin = "*" * ((78 - len(title))//2)
    print("\n")
    print("=" * 80)
    print(f"{margin} {title} {margin}")
    print("=" * 80)

def preprocess(image):
    """We don't transform the validation/testing data from augmentation pov, 
    only training data exclusively gets transformed.
    However, we need to resize and convert the testing data as per resnet architecture."""
    test_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(size=(IMAGE.height, IMAGE.width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return test_transform(image)

def get_food_class(indices):
    classes =  ['biryani', 'cholebhature', 'dabeli', 'dal', 'dhokla', 
                'dosa', 'jalebi', 'kathiroll', 'kofta', 'naan', 
                'pakora', 'paneer', 'panipuri', 'pavbhaji', 'vadapav']

    return [classes[index] for index in indices]