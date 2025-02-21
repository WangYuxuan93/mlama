# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from .bert_connector import Bert
from .bert_connector2 import Bert2
from .xlm_roberta_connector import XLMRoberta
from .xlm_roberta_connector import MEAE

# bert is loaded with Bert from pytorch_pretrained_bert
# bert2 is loaded with Bert from transformers


def build_model_by_name(lm, args, verbose=True):
    """Load a model by name and args.

    Note, args.lm is not used for model selection. args are only passed to the
    model's initializator.
    """
    MODEL_NAME_TO_CLASS = dict(
        bert=Bert,
        bert2=Bert2,
        xlmr=XLMRoberta,
        meae=MEAE,
    )
    if lm not in MODEL_NAME_TO_CLASS:
        raise ValueError("Unrecognized Language Model: %s." % lm)
    if verbose:
        print("Loading %s model..." % lm)
    return MODEL_NAME_TO_CLASS[lm](args)
