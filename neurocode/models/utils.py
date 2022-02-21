"""
Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 21-02-2022
"""
import torch

def load_model(paramspath):
    try:
        model = torch.load(paramspath)
    except Exception as e:
        print(
            f'Could not load model at path {paramspath}')
        print(e)

    return model
    