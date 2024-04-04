1.2.0 (2022-05-17)
======================

### Features

- Add support for GridSampler ([#1815](https://github.com/facebookresearch/hydra/issues/1815))
- Support for Python 3.10 ([#1856](https://github.com/facebookresearch/hydra/issues/1856))
- Add experimental 'custom_search_space' configuration node to allow extending trial objects programmatically. ([#1906](https://github.com/facebookresearch/hydra/issues/1906))

### Configuration structure changes

- Add hydra.sweeper.params and deprecate hydra.sweeper.search_space ([#1890](https://github.com/facebookresearch/hydra/issues/1890))


1.1.2 (2022-01-23)
=======================

### Bug Fixes

- Fix a bug where Optuna Sweeper parses the override value incorrectly ([#1811](https://github.com/facebookresearch/hydra/issues/1811))


1.1.1 (2021-09-01)
=======================

### Maintenance Changes

- Update optuna dependency ([#1746](https://github.com/facebookresearch/hydra/issues/1634))


1.1.0.dev2 (2021-06-10)
=======================

### Features

- Add support for changing settings of Optuna samplers ([#1472](https://github.com/facebookresearch/hydra/issues/1472))

### API Change (Renames, deprecations and removals)

- Config structure changes, please refer to the [docs](https://hydra.cc/docs/plugins/optuna_sweeper/) ([#1472](https://github.com/facebookresearch/hydra/issues/1472))
