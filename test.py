import torch
from model.Net import CoarseNet,FineNet
import os
from torchvision import transforms
from PIL import Image
import conf
cuda = True if torch.cuda.is_available() else False


def Test_demo():
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    coarse_net = CoarseNet().to(conf.device)
    coarse_net.load_state_dict(torch.load(conf.coarse_model_path))  # , False
    coarse_net.eval()
    fine_net = FineNet().to(conf.device)
    fine_net.load_state_dict(torch.load(conf.fine_model_path))  # , False
    fine_net.eval()

    img = Image.open(conf.test_path)
    img_tensor = transforms.ToTensor()(img)
    img_tensor = img_tensor.to(conf.device)
    img_tensor = img_tensor.unsqueeze(0)
    coarse, _, _ = coarse_net(img_tensor)
    res = fine_net(coarse.detach())
    res = transforms.ToPILImage()(res.cpu().squeeze(0))
    res.save(conf.result_path)


if __name__ == '__main__':
    Test_demo()