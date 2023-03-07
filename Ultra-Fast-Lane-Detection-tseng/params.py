import torch
# from torchsummary import summary
from torchinfo import summary
import torchvision
from model.model import parsingNet
# torch.backends.cudnn.deterministic = False

torch.backends.cudnn.benchmark = True
net = parsingNet(pretrained = False, backbone='18p',cls_dim = (100+1,56,4),use_aux=False).cuda()
# cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device =torch.device("cpu")
if __name__ == "__main__":
    summary(net, (1, 3, 288, 800))
