# SPDX-License-Identifier: Apache-2.0

import warnings

import hydra

warnings.filterwarnings("ignore")


@hydra.main(config_path="conf/", config_name="inference_config")
def main(cfg):
    print(cfg)
    cfg._target_ = cfg.config.inference._target_
    hydra.utils.call(cfg)


if __name__ == "__main__":
    main()
