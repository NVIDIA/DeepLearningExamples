from omegaconf import OmegaConf, open_dict

from data.data_utils import DataTypes, InputTypes, translate_features


def append_derived_config_fields(config):
    OmegaConf.set_struct(config, False)
    config = config.config
    features = translate_features(config.dataset.features)
    with open_dict(config):
        config.model.example_length = config.dataset.example_length
        config.model.encoder_length = config.dataset.encoder_length
        config.model.temporal_known_continuous_inp_size = len(
            [x for x in features if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CONTINUOUS]
        )
        config.model.temporal_observed_continuous_inp_size = len(
            [x for x in features if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CONTINUOUS]
        )
        config.model.temporal_target_size = len([x for x in features if x.feature_type == InputTypes.TARGET])
        config.model.static_continuous_inp_size = len(
            [x for x in features if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CONTINUOUS]
        )
        config.model.static_categorical_inp_lens = [
            # XXX: this might be a bad idea. It is better make cardinality required.
            x.get("cardinality", 100)
            for x in features
            if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CATEGORICAL
        ]

        config.model.temporal_known_categorical_inp_lens = [
            x.get("cardinality", 100)
            for x in features
            if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CATEGORICAL
        ]
        config.model.temporal_observed_categorical_inp_lens = [
            x.get("cardinality", 100)
            for x in features
            if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CATEGORICAL
        ]

        config.model.num_static_vars = config.model.static_continuous_inp_size + len(
            config.model.static_categorical_inp_lens
        )
        config.model.num_future_vars = config.model.temporal_known_continuous_inp_size + len(
            config.model.temporal_known_categorical_inp_lens
        )
        config.model.num_historic_vars = sum(
            [
                config.model.num_future_vars,
                config.model.temporal_observed_continuous_inp_size,
                config.model.temporal_target_size,
                len(config.model.temporal_observed_categorical_inp_lens),
            ]
        )
