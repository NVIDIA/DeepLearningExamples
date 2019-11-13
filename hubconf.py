def relocated():
    raise ValueError(
        "NVIDIA entrypoints moved to branch torchhub \n"
        "Use torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', ...) to access the models"
    )


def nvidia_ncf(**kwargs):
    """Entrypoints moved to branch torchhub
    """
    relocated()


def nvidia_tacotron2(**kwargs):
    """Entrypoints moved to branch torchhub
    """
    relocated()


def nvidia_waveglow(**kwargs):
    """Entrypoints moved to branch torchhub
    """
    relocated()


def nvidia_ssd_processing_utils():
    """Entrypoints moved to branch torchhub
    """
    relocated()


def nvidia_ssd(**kwargs):
    """Entrypoints moved to branch torchhub
    """
    relocated()
