from engines.training import normal_trainer, hie_trainer
from engines.validating import normal_validator, hie_validator
from engines.testing import normal_tester, hie_tester


def call_engines(args):
    if "hie" in args.model_name.lower():
        return hie_trainer, hie_validator, hie_tester
    else:
        return normal_trainer, normal_validator, normal_tester