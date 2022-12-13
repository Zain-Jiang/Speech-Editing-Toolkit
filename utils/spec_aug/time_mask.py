import random
import numpy as np
import torch
import torch.nn.functional as F

def generate_time_mask(
    spec: torch.Tensor,
    # max_length: int = 100,
    ratio: 0.1,
    num_mask: int = 1,
    replace_with_zero: bool = True,
):
    """Apply mask along the specified direction.
    Args:
        spec: (Length, Freq)
        spec_lengths: (Length): Not using lengths in this implementation
        mask_width_range: Select the width randomly between this range
    """

    org_size = spec.size()

    # D = Length
    D = spec.shape[0]
    # # mask_length: (num_mask)
    # if int(D*0.10) < int(D*0.14):
    #     mask_width_range = [int(D*0.10), int(D*0.14)]
    # else:
    #     mask_width_range = [int(D*0.10), int(D*0.14)+1]
    # mask_length = torch.randint(
    #     mask_width_range[0],
    #     mask_width_range[1],
    #     (num_mask,1),
    #     device=spec.device,
    # )
    mask_length = int(D*ratio)

    # mask_pos: (num_mask)
    mask_pos = torch.randint(
        0, max(1, D - mask_length), (num_mask,1), device=spec.device
    )

    # aran: (1, D)
    aran = torch.arange(D, device=spec.device)[None, :]
    # spec_mask: (num_mask, D)
    spec_mask = (mask_pos <= aran) * (aran < (mask_pos + mask_length))
    # Multiply masks: (num_mask, D) -> (D)
    spec_mask = spec_mask.any(dim=0).float()
    return spec_mask

def generate_alignment_aware_time_mask(
    spec: torch.Tensor,
    mel2ph,
    # max_length: int = 100,
    ratio: 0.1,
    num_mask: int = 1,
    replace_with_zero: bool = True,
):
    # obtain ph-level mask
    ph_mask = np.zeros((mel2ph.max()+1).item())
    ph_seq_idx = np.arange(0, mel2ph.max(), dtype=float) # start from 1 to match the mel2ph
    mask_ph_idx = np.random.choice(ph_seq_idx, size=int((mel2ph.max()+1)*ratio), replace=False).astype(np.uint8)
    ph_mask[mask_ph_idx] = 1.0
    ph_mask = torch.from_numpy(ph_mask).float()

    # obtain mel-level mask
    ph_mask = F.pad(ph_mask, [1, 0])
    mel2ph_ = mel2ph
    mel_mask = torch.gather(ph_mask, 0, mel2ph_)  # [B, T, H]

    return mel_mask

def generate_inference_mask(
    spec: torch.Tensor,
    mel2ph,
    # max_length: int = 100,
    ratio: 0.3,
    num_mask: int = 1,
    replace_with_zero: bool = True,
):
    # obtain ph-level mask
    ph_mask = np.zeros((mel2ph.max()+1).item())
    ph_seq_idx = np.arange(0, mel2ph.max(), dtype=float) # start from 1 to match the mel2ph
    mask_ph_idx = random.randint(0, int(mel2ph.max() - mel2ph.max() * ratio))

    ph_mask[mask_ph_idx: int(mask_ph_idx + mel2ph.max() * ratio)] = 1.0
    ph_mask = torch.from_numpy(ph_mask).float()

    # obtain mel-level mask
    ph_mask = F.pad(ph_mask, [1, 0])
    mel2ph_ = mel2ph
    mel_mask = torch.gather(ph_mask, 0, mel2ph_)  # [B, T, H]

    return mel_mask