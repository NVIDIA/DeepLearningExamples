"""
Returns Unet3+ model
"""
import tensorflow as tf
from omegaconf import DictConfig

from .backbones import vgg16_backbone, vgg19_backbone, unet3plus_backbone
from .unet3plus import unet3plus
from .unet3plus_deep_supervision import unet3plus_deepsup
from .unet3plus_deep_supervision_cgm import unet3plus_deepsup_cgm


def prepare_model(cfg: DictConfig, training=False):
    """
    Creates and return model object based on given model type.
    """

    input_shape = [cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH, cfg.INPUT.CHANNELS]
    input_layer = tf.keras.layers.Input(
        shape=input_shape,
        name="input_layer"
    )  # 320*320*3
    filters = [64, 128, 256, 512, 1024]

    #  create backbone
    if cfg.MODEL.BACKBONE.TYPE == "unet3plus":
        backbone_layers = unet3plus_backbone(
            input_layer,
            filters
        )
    elif cfg.MODEL.BACKBONE.TYPE == "vgg16":
        backbone_layers = vgg16_backbone(input_layer, )
    elif cfg.MODEL.BACKBONE.TYPE == "vgg19":
        backbone_layers = vgg19_backbone(input_layer, )
    else:
        raise ValueError(
            "Wrong backbone type passed."
            "\nPlease check config file for possible options."
        )
    print(f"Using {cfg.MODEL.BACKBONE.TYPE} as a backbone.")

    if cfg.MODEL.TYPE == "unet3plus":
        #  training parameter does not matter in this case
        outputs, model_name = unet3plus(
            backbone_layers,
            cfg.OUTPUT.CLASSES,
            filters
        )
    elif cfg.MODEL.TYPE == "unet3plus_deepsup":
        outputs, model_name = unet3plus_deepsup(
            backbone_layers,
            cfg.OUTPUT.CLASSES,
            filters,
            training
        )
    elif cfg.MODEL.TYPE == "unet3plus_deepsup_cgm":
        if cfg.OUTPUT.CLASSES != 1:
            raise ValueError(
                "UNet3+ with Deep Supervision and Classification Guided Module"
                "\nOnly works when model output classes are equal to 1"
            )
        outputs, model_name = unet3plus_deepsup_cgm(
            backbone_layers,
            cfg.OUTPUT.CLASSES,
            filters,
            training
        )
    else:
        raise ValueError(
            "Wrong model type passed."
            "\nPlease check config file for possible options."
        )

    return tf.keras.Model(
        inputs=input_layer,
        outputs=outputs,
        name=model_name
    )


if __name__ == "__main__":
    """## Test model Compilation,"""
    from omegaconf import OmegaConf

    cfg = {
        "WORK_DIR": "H:\\Projects\\UNet3P",
        "INPUT": {"HEIGHT": 320, "WIDTH": 320, "CHANNELS": 3},
        "OUTPUT": {"CLASSES": 1},
        # available variants are unet3plus, unet3plus_deepsup, unet3plus_deepsup_cgm
        "MODEL": {"TYPE": "unet3plus",
                  # available variants are unet3plus, vgg16, vgg19
                  "BACKBONE": {"TYPE": "vgg19", }
                  }
    }
    unet_3P = prepare_model(OmegaConf.create(cfg), True)
    unet_3P.summary()

    # tf.keras.utils.plot_model(unet_3P, show_layer_names=True, show_shapes=True)

    # unet_3P.save("unet_3P.hdf5")
