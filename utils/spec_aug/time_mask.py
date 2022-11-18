import torch
from typing import Sequence

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