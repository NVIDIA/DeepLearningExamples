import os
import sys

from PyTorch.Detection.SSD.ssd import nvidia_ssd, nvidia_ssd_processing_utils
sys.path.append(os.path.join(sys.path[0], 'PyTorch/Detection/SSD'))

from PyTorch.Classification.ConvNets.image_classification.models import resnet50 as nvidia_resnet50
from PyTorch.Classification.ConvNets.image_classification.models import resnext101_32x4d as nvidia_resnext101_32x4d
from PyTorch.Classification.ConvNets.image_classification.models import se_resnext101_32x4d as nvidia_se_resnext101_32x4d
from PyTorch.Classification.ConvNets.image_classification.models import efficientnet_b0 as nvidia_efficientnet_b0
from PyTorch.Classification.ConvNets.image_classification.models import efficientnet_b4 as nvidia_efficientnet_b4
from PyTorch.Classification.ConvNets.image_classification.models import efficientnet_widese_b0 as nvidia_efficientnet_widese_b0
from PyTorch.Classification.ConvNets.image_classification.models import efficientnet_widese_b4 as nvidia_efficientnet_widese_b4
from PyTorch.Classification.ConvNets.image_classification.models import nvidia_convnets_processing_utils

from PyTorch.Classification.ConvNets.image_classification.models import resnext101_32x4d as nvidia_resneXt
from PyTorch.Classification.ConvNets.image_classification.models import nvidia_efficientnet
sys.path.append(os.path.join(sys.path[0], 'PyTorch/Classification/ConvNets/image_classification'))

from PyTorch.Classification.GPUNet.configs.gpunet_torchhub import nvidia_gpunet
sys.path.append(os.path.join(sys.path[0], 'PyTorch/Classification/GPUNet/configs'))

from PyTorch.SpeechSynthesis.Tacotron2.tacotron2 import nvidia_tacotron2
from PyTorch.SpeechSynthesis.Tacotron2.tacotron2 import nvidia_tts_utils
from PyTorch.SpeechSynthesis.Tacotron2.waveglow import nvidia_waveglow
sys.path.append(os.path.join(sys.path[0], 'PyTorch/SpeechSynthesis/Tacotron2'))

from PyTorch.SpeechSynthesis.HiFiGAN.fastpitch import nvidia_fastpitch
from PyTorch.SpeechSynthesis.HiFiGAN.fastpitch import nvidia_textprocessing_utils
from PyTorch.SpeechSynthesis.HiFiGAN.hifigan import nvidia_hifigan
sys.path.append(os.path.join(sys.path[0], 'PyTorch/SpeechSynthesis/HiFiGAN'))

from PyTorch.Forecasting.TFT.tft_torchhub import nvidia_tft, nvidia_tft_data_utils
sys.path.append(os.path.join(sys.path[0], 'PyTorch/Forecasting/TFT'))
