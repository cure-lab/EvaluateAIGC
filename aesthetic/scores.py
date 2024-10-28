from torchvision import transforms
from aesthetic.model import AestheticModel
import torch

def get_aesthetic_scores(image, clip_feature):
    image = image.resize([256,256]).convert("RGB")
    opts = {"depth":4,"num_attributes":6,"num_distortions":10}

    overall_model = AestheticModel(opts)
    overall_model.load_state_dict(torch.load("./ckpt/aesthetic/score.pth"))
    overall_model = overall_model.cuda()
    overall_model.eval()

    attribute_model = AestheticModel(opts)
    attribute_model.load_state_dict(torch.load("./ckpt/aesthetic/attribute.pth"))
    attribute_model = attribute_model.cuda()
    attribute_model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    attribute_names = ["content", "light", "dof", "emphasis","color","composition"]
    with torch.no_grad():
        image = transform(image).unsqueeze(0).cuda()
        _, attribute_scores = attribute_model(image, clip_feature)
        scores, _ = overall_model(image, clip_feature)
        score_dict = {attribute:float(value.cpu()) for attribute, value in zip(attribute_names, attribute_scores[0])}
        score_dict["overall"] = float(scores[0][0].cpu())
        score_dict["emphasis"] = score_dict["emphasis"] * 10
    return score_dict