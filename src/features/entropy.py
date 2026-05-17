import torch


def compute_feature_entropy(feature_tensor):
    """
    Improved entropy computation.

    Computes entropy channel-wise
    then averages across channels.

    Args:
        feature_tensor: torch.Tensor
            Shape: [B, C, H, W]

    Returns:
        float
    """

    # Remove batch dimension
    x = feature_tensor.squeeze(0)

    # Shape:
    # [C, H, W]

    channel_entropies = []

    for idx, channel in enumerate(x):

        # Flatten spatial map
        c = channel.flatten()

        # Convert to positive distribution
        c = torch.abs(c)

        # Normalize
        c = c / (torch.sum(c) + 1e-8)

        # Shannon entropy
        entropy = -torch.sum(c * torch.log(c + 1e-8))

        channel_entropies.append(entropy)

    # Average entropy across channels
    final_entropy = torch.mean(torch.stack(channel_entropies))

    return final_entropy.item()