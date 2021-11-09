import os
import sys

from PyTorch.Detection.SSD.src import nvidia_ssd, nvidia_ssd_processing_utils
sys.path.append(os.path.join(sys.path[0], 'PyTorch/Detection/SSD'))

from PyTorch.Classification.ConvNets.image_classification.models import nvidia_efficientnet, nvidia_resneXt, nvidia_resnet50, nvidia_convnets_processing_utils
sys.path.append(os.path.join(sys.path[0], 'PyTorch/Classification/ConvNets/image_classification'))

from PyTorch.SpeechSynthesis.Tacotron2.tacotron2 import nvidia_tacotron2
from PyTorch.SpeechSynthesis.Tacotron2.tacotron2 import nvidia_tts_utils
from PyTorch.SpeechSynthesis.Tacotron2.waveglow import nvidia_waveglow
sys.path.append(os.path.join(sys.path[0], 'PyTorch/SpeechSynthesis/Tacotron2'))
