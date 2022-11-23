import torch.nn as nn

class MelEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_size=192):
        super(MelEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # Linear function (out)
        self.fc_out = nn.Linear(hidden_size, hidden_size)  

    def forward(self, x):
        out = self.encoder(x)
        # Linear function (out)
        out = self.fc_out(out)
        return out