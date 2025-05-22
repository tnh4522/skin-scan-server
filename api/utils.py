import torch
import numpy as np
from torchvision import transforms


def load_checkpoint(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    state_dict = state_dict["model"] if "model" in state_dict else state_dict
    model.load_state_dict(state_dict)
    return model


def preprocess_image(image, texture, size=1024):
    image = image.resize((size, size))
    texture = texture.resize((size, size))

    image = np.array(image).astype(np.float32).transpose(2, 0, 1)
    texture = np.array(texture).astype(np.float32)[None, ...]

    image = image / 255.0 * 2.0 - 1.0
    texture = texture / 255.0 * 2.0 - 1.0

    combined = np.concatenate([image, texture], axis=0)
    return torch.tensor(combined, dtype=torch.float32)