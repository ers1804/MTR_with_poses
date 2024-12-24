# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


from .mtr_encoder import MTREncoder, JEPAEncoder, JEPATransformerEncoder

__all__ = {
    'MTREncoder': MTREncoder,
    'JEPAEncoder': JEPAEncoder,
    'JEPATransformerEncoder': JEPATransformerEncoder,
}


def build_context_encoder(config):
    model = __all__[config.NAME](
        config=config
    )

    return model
