from numpy import isin
from scipy.misc import face
import torch
from torch import exp_, nn
from torch.nn import functional as F


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class SingleWindowDisc(nn.Module):
    def __init__(self, time_length, freq_length=80, hidden_size=256, kernel=(3, 3), c_in=1):
        super(SingleWindowDisc, self).__init__()

        # mel = torch.rand(B, 1, 80, t=10)
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.model = nn.ModuleList([
            nn.Sequential(*[
                nn.Conv2d(c_in, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.BatchNorm2d(hidden_size, 0.8)
            ]),
            nn.Sequential(*[
                nn.Conv2d(hidden_size, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.BatchNorm2d(hidden_size, 0.8)
            ]),         
            nn.Sequential(*[
                nn.Conv2d(hidden_size, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]),
        ])
        ds_size = (time_length // 2 ** 3, (freq_length + 7) // 2 ** 3)
        self.out_layer = nn.Linear(hidden_size * ds_size[0] * ds_size[1], 1)

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x): 
        """
        :param x: [B, C, T, n_bins]
        :return: validity: [B, 1], h: List of hiddens
        """
        for l in self.model:
            x = l(x)
        x = x.view(x.shape[0], -1)
        x = self.out_layer(x)  # [B, 1]
        return x