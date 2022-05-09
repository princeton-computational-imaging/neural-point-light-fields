import torch
import torch.nn as nn
# from all_utils import DATASET_NUM_CLASS
from .simpleview_utils import PCViews, Squeeze, BatchNormPoint
import matplotlib.pyplot as plt

from .resnet import _resnet, BasicBlock


class MVModel(nn.Module):
    def __init__(self, task,
                 # dataset,
                 backbone,
                 feat_size,
                 resolution=128,
                 upscale_feats = True,
                 ):
        super().__init__()
        assert task == 'cls'
        self.task = task
        self.num_class = 10 # DATASET_NUM_CLASS[dataset]
        self.dropout_p = 0.5
        self.feat_size = feat_size

        self.resolution = resolution
        self.upscale_feats = False

        pc_views = PCViews()
        self.num_views = pc_views.num_views
        self._get_img = pc_views.get_img
        self._get_img_coord = pc_views.get_img_coord

        img_layers, in_features = self.get_img_layers(
            backbone, feat_size=feat_size)
        self.img_model = nn.Sequential(*img_layers)

        # self.final_fc = MVFC(
        #     num_views=self.num_views,
        #     in_features=in_features,
        #     out_features=self.num_class,
        #     dropout_p=self.dropout_p)

        # Upscale resnet outputs to img resolution
        if upscale_feats:
            self.upscale_feats = True

            self.upscaleLin = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(in_features=self.feat_size * (2 ** i), out_features=self.resolution), nn.ReLU()
                    ) for i in range(4)
                ]
            )
            self.upsampleLayer = nn.Upsample(size=(resolution, resolution), mode='nearest')

            self.out_idx = [3, 4, 5, 7]

    def forward(self, pc, **kwargs):
        """
        :param pc:
        :return:
        """

        # Does not give the same results if trained on a single or more images, because of batch norm
        pc = pc.cuda()
        img = self.get_img(pc)

        # feat = self.img_model(img)
        outs = []
        h = img
        for layer in self.img_model:
            h = layer(h)
            outs.append(h)

        if self.upscale_feats:
            feat_ls = []
            for i, upLayer in zip(self.out_idx, self.upscaleLin):
                h = outs[i].transpose(1,-1)
                h = upLayer(h).transpose(-1,1)
                feat_ls.append(self.upsampleLayer(h))
            feat = torch.sum(torch.stack(feat_ls), dim=0) / len(self.upscaleLin)

            # plt.imshow(torch.sum(outs[7].transpose(1,-1), dim=-1).cpu().detach().numpy()[0])
            # plt.imshow(torch.sum(feat.transpose(1, -1), dim=-1).cpu().detach().numpy()[0])
        else:
            feat = outs[-1]

        # if len(pc) >= 1:
        #     i_sh = [6, len(pc)] + list(img.shape[1:])
        #     f_sh = [6, len(pc)] + list(feat.shape[1:])
        #     img = img.reshape(i_sh)
        #     feat = feat.reshape(f_sh)
        #
        # else:
        #     img = img.unsqueeze(1)
        #     feat = feat.unsqueeze(1)

        n_img, in_ch, w_in, h_in = img.shape
        n_feat_maps, out_ch, w_out, h_out = feat.shape
        n_batch = len(pc)
        feat = feat.reshape(n_batch, n_feat_maps//n_batch, out_ch, w_out, h_out)
        img = img.reshape(n_batch, n_img // n_batch, in_ch, w_in, h_in)

        return feat, img, None


    def get_img(self, pc):
        img = self._get_img(pc, self.resolution)
        img = torch.tensor(img).float()
        img = img.to(next(self.parameters()).device)
        assert len(img.shape) == 3
        img = img.unsqueeze(3)
        # [num_pc * num_views, 1, RESOLUTION, RESOLUTION]
        img = img.permute(0, 3, 1, 2)

        return img

    @staticmethod
    def get_img_layers(backbone, feat_size):
        """
        Return layers for the image model
        """
        assert backbone == 'resnet18'
        layers = [2, 2, 2, 2]
        block = BasicBlock
        backbone_mod = _resnet(
            arch=None,
            block=block,
            layers=layers,
            pretrained=False,
            progress=False,
            feature_size=feat_size,
            zero_init_residual=True)

        all_layers = [x for x in backbone_mod.children()]
        in_features = all_layers[-1].in_features

        # all layers except the final fc layer.py and the initial conv layers
        # WARNING: this is checked only for resnet models

        # main_layers = all_layers[4:-1]
        main_layers = all_layers[4:-2]
        img_layers = [
            nn.Conv2d(1, feat_size, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(feat_size, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            *main_layers,
            Squeeze()
        ]

        return img_layers, in_features


class MVFC(nn.Module):
    """
    Final FC layers for the MV model
    """

    def __init__(self, num_views, in_features, out_features, dropout_p):
        super().__init__()
        self.num_views = num_views
        self.in_features = in_features
        self.model = nn.Sequential(
            BatchNormPoint(in_features),
            # dropout before concatenation so that each view drops features independently
            nn.Dropout(dropout_p),
            nn.Flatten(),
            nn.Linear(in_features=in_features * self.num_views,
                      out_features=in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(in_features=in_features, out_features=out_features,
                      bias=True))

    def forward(self, feat):
        feat = feat.view((-1, self.num_views, self.in_features))
        out = self.model(feat)
        return out
