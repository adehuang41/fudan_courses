"""Model factory: ResNet-{18,34} (+SE/CBAM) and timm ViT/Swin."""
import torch.nn as nn
from .resnet_attn import build_resnet


def build_model(name: str, pretrained: bool = True, num_classes: int = 37,
                identity_init: bool = False):
    """name examples:
        resnet18, resnet34
        resnet18_se, resnet18_cbam
        vit_tiny, swin_tiny
    """
    name = name.lower()
    if name == "resnet18":
        return build_resnet(18, pretrained, num_classes, attn="none")
    if name == "resnet34":
        return build_resnet(34, pretrained, num_classes, attn="none")
    if name == "resnet18_se":
        return build_resnet(18, pretrained, num_classes, attn="se",
                            identity_init=identity_init)
    if name == "resnet18_cbam":
        return build_resnet(18, pretrained, num_classes, attn="cbam",
                            identity_init=identity_init)
    if name in ("vit_tiny", "vit_tiny_patch16_224"):
        import timm
        return timm.create_model("vit_tiny_patch16_224", pretrained=pretrained,
                                 num_classes=num_classes)
    if name in ("seresnet18_timm", "legacy_seresnet18"):
        import timm
        return timm.create_model("legacy_seresnet18", pretrained=pretrained,
                                 num_classes=num_classes)
    if name in ("swin_tiny", "swin_tiny_patch4_window7_224"):
        import timm
        return timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained,
                                 num_classes=num_classes)
    raise ValueError(f"unknown model: {name}")


def split_params(model: nn.Module, name: str):
    """Return (head_params, backbone_params, attn_params).
    head = final classifier; attn = SE/CBAM submodules; backbone = the rest.
    """
    name = name.lower()
    if hasattr(model, "get_classifier"):
        head_ids = set(id(p) for p in model.get_classifier().parameters())
    elif name.startswith("resnet"):
        head_ids = set(id(p) for p in model.fc.parameters())
    elif name.startswith("vit") or name.startswith("swin"):
        head_ids = set(id(p) for p in model.head.parameters())
    else:
        head_ids = set()
    attn_ids = set()
    for mod_name, mod in model.named_modules():
        if mod_name.endswith(".attn") and len(list(mod.parameters())) > 0:
            for p in mod.parameters():
                attn_ids.add(id(p))
    head, bb, at = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            head.append(p)
        elif id(p) in attn_ids:
            at.append(p)
        else:
            bb.append(p)
    return head, bb, at
