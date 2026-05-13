"""ResNet-18/34 with optional SE-block or CBAM inserted into BasicBlock."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, identity_init: bool = False):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)
        if identity_init:
            nn.init.zeros_(self.fc2.weight)
            nn.init.constant_(self.fc2.bias, 4.0)  # sigmoid(4) ~= 0.982

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7,
                 identity_init: bool = False):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.spatial = nn.Conv2d(2, 1, spatial_kernel, padding=spatial_kernel // 2, bias=True)
        if identity_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.constant_(self.mlp[-1].bias, 2.0)  # sigmoid(2+2)=sigmoid(4)~=0.982 after avg+max
            nn.init.zeros_(self.spatial.weight)
            nn.init.constant_(self.spatial.bias, 4.0)

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        ca = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        x = x * ca
        avg_s = x.mean(dim=1, keepdim=True)
        max_s, _ = x.max(dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial(torch.cat([avg_s, max_s], dim=1)))
        return x * sa


class AttnBasicBlock(BasicBlock):
    """BasicBlock with attention applied to bn2 output, before residual add."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 attn: str = "none", identity_init: bool = False):
        super().__init__(inplanes, planes, stride, downsample, groups, base_width,
                         dilation, norm_layer)
        if attn == "se":
            self.attn = SEBlock(planes * self.expansion, identity_init=identity_init)
        elif attn == "cbam":
            self.attn = CBAM(planes * self.expansion, identity_init=identity_init)
        else:
            self.attn = nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attn(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def _make_attn_resnet(depth: int, attn: str, num_classes: int, pretrained: bool,
                      identity_init: bool = False):
    layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3]}[depth]
    attn_kind = attn
    id_init = identity_init

    class _Block(AttnBasicBlock):
        expansion = 1
        def __init__(self, *args, **kwargs):
            super().__init__(*args, attn=attn_kind, identity_init=id_init, **kwargs)

    model = ResNet(_Block, layers, num_classes=1000)
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1 if depth == 18 else ResNet34_Weights.IMAGENET1K_V1
        sd = weights.get_state_dict(progress=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # attn submodules are expected to be missing; fc will be replaced anyway
        for k in missing:
            assert ("attn" in k) or k.startswith("fc."), f"unexpected missing key {k}"
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    nn.init.kaiming_normal_(model.fc.weight, nonlinearity="linear")
    nn.init.zeros_(model.fc.bias)
    return model


def build_resnet(depth: int = 18, pretrained: bool = True,
                 num_classes: int = 37, attn: str = "none",
                 identity_init: bool = False):
    if attn in ("se", "cbam"):
        return _make_attn_resnet(depth, attn, num_classes, pretrained,
                                 identity_init=identity_init)
    weights = None
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1 if depth == 18 else ResNet34_Weights.IMAGENET1K_V1
    fn = resnet18 if depth == 18 else resnet34
    model = fn(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    nn.init.kaiming_normal_(model.fc.weight, nonlinearity="linear")
    nn.init.zeros_(model.fc.bias)
    return model
