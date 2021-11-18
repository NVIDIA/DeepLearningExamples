# SPDX-License-Identifier: Apache-2.0

import warnings

import hydra

warnings.filterwarnings("ignore")


@hydra.main(config_path="conf/", config_name="preproc_config")
def main(cfg):
    print(cfg)
    hydra.utils.call(cfg)


if __name__ == "__main__":
    main()
