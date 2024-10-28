import open_clip
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


model, vis_processors, txt_processors = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
model = model.cuda()

class ClipDataset(Dataset):
    def __init__(self, data):
        self.imgs = data
        self.clip_trans = vis_processors

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image = Image.open(self.imgs[index]).convert("RGB")
        clip_image = self.clip_trans(image)
        return clip_image

def get_clip_features(opts, paths):
    data_loader = DataLoader(ClipDataset(paths), pin_memory=True, 
                shuffle=False, batch_size=opts["batch_size"], num_workers=opts["num_workers"], drop_last=False)
    all_features = []
    with torch.no_grad():
        for clip_data in data_loader:
            clip_feature = model.encode_image(clip_data.cuda())
            all_features.append(clip_feature.detach())
    all_features = torch.concat(all_features, dim=0)
    return all_features

def get_clip_feature(image):
    image = vis_processors(image).unsqueeze(0)
    with torch.no_grad():
        feature = model.encode_image(image.cuda())
    return feature