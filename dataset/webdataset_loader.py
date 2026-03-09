import webdataset as wds
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

def create_loader(urls,batch_size):

    dataset = (
        wds.WebDataset(urls)
        .shuffle(1000)
        .decode("pil")
        .to_tuple("jpg","txt")
        .map_tuple(transform, lambda x:x)
        .batched(batch_size)
    )

    return dataset