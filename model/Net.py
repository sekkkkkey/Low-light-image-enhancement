import torch
import torch.nn as nn
import conf
channels = conf.channel
channels_2 = conf.channel_d
illu_channel = conf.illu_channel
rgb_channel = conf.rgb_channel
contact_channel = conf.contact_channel
eps = conf.eps
class CoarseNet(nn.Module):
    def __init__(self):
        super(CoarseNet, self).__init__()
        self.noise_net = nn.Sequential(
            nn.Conv2d(rgb_channel, channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels_2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels_2, channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels, rgb_channel, 3, 1, 1),
        )
        self.illu_net = nn.Sequential(
            nn.Conv2d(illu_channel, channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels_2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels_2, channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels, illu_channel, 3, 1, 1),
        )


    def forward(self,img_tensor):

        illu_in = torch.max(img_tensor,dim=1)[0].unsqueeze(1)
        illu = torch.sigmoid(self.illu_net(illu_in))
        noise = torch.tanh(self.noise_net(img_tensor))
        I_res = (img_tensor-noise)/(illu+eps)
        I_res = torch.clamp(I_res,min=0,max=1)
        return  I_res,illu,noise

class FineNet(nn.Module):
        def __init__(self):
            super(FineNet, self).__init__()
            self.FEN1_net1 = nn.Sequential(
                nn.Conv2d(rgb_channel, channels, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.ReLU(),
            )
            self.FEN1_net2 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.ReLU(),
            )

            self.FEN2_net1 = nn.Sequential(
                nn.Conv2d(channels, channels_2, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(channels_2, channels, 3, 1, 1),
                nn.ReLU(),
            )

            self.DEN = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(channels, rgb_channel, 3, 1, 1),
                nn.ReLU(),
            )

            self.CAN = nn.Sequential(
                nn.Conv2d(contact_channel, channels, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(channels, channels_2, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(channels_2, channels, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(channels, rgb_channel, 3, 1, 1),
                nn.ReLU(),
            )



        def forward(self, I_res):
            DAB = dual_ateention().to('cuda')
            F1 = self.FEN1_net1(I_res)
            F2 = self.FEN1_net2(F1)
            F3 = self.FEN1_net2(F2)
            F4 = self.FEN1_net2(F3)
            FEN_F1 = self.FEN2_net1(F1)
            FEN_F2 = self.FEN2_net1(F2)
            FEN_F3 = self.FEN2_net1(F3)
            FEN_F4 = self.FEN2_net1(F4)
            F1_c = DAB(F1,FEN_F1)
            F2_c = DAB(F2,FEN_F2)
            F3_c = DAB(F3,FEN_F3)
            F4_c = DAB(F4,FEN_F4)
            DEN_F1 = self.DEN(F1_c)
            DEN_F2 = self.DEN(F2_c)
            DEN_F3 = self.DEN(F3_c)
            DEN_F4 = self.DEN(F4_c)
            F1_c2 = torch.cat([I_res, DEN_F1], dim=1)
            I1 = self.CAN(F1_c2)
            F2_c2 = torch.cat([I1, DEN_F2], dim=1)
            I2 = self.CAN(F2_c2)
            F3_c2 = torch.cat([I2, DEN_F3], dim=1)
            I3 = self.CAN(F3_c2)
            F4_c2 = torch.cat([I3, DEN_F4], dim=1)
            res = self.CAN(F4_c2)
            return res


class dual_ateention(nn.Module):
    def __init__(self):
        super(dual_ateention, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(channels, channels_2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels_2, channels, 3, 1, 1),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels_2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels_2, channels, 3, 1, 1),
        )

    def forward(self, x,FEN_F1):
        x1 = torch.mean(torch.mean(x,dim=3),dim=2).unsqueeze(2).unsqueeze(3)
        x2 = torch.sigmoid(self.conv_1(x1))
        x3 = x2*x1
        x4 = torch.sigmoid(self.conv_2(x3))
        res = (x3*x4)*FEN_F1
        return res
