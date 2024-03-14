from engines.FSDP_training import normal_trainer  as FSDP_normal_trainer
from engines.FSDP_training import hie_trainer as FSDP_hie_trainer
from engines.FSDP_validating import normal_validator as FSDP_normal_validator
from engines.FSDP_validating import  hie_validator as FSDP_hie_validator
from engines.FSDP_testing import normal_tester as FSDP_normal_tester
from engines.FSDP_testing import hie_tester as  FSDP_hie_tester

from engines.DDP_training import normal_trainer  as DDP_normal_trainer
from engines.DDP_training import hie_trainer as DDP_hie_trainer
from engines.DDP_validating import normal_validator as DDP_normal_validator
from engines.DDP_validating import  hie_validator as DDP_hie_validator
from engines.DDP_testing import normal_tester as DDP_normal_tester
from engines.DDP_testing import hie_tester as  DDP_hie_tester

def fsdp_call_engines(args):
    if "hie" in args.model_name.lower():
        return FSDP_hie_trainer, FSDP_hie_validator, FSDP_hie_tester
    else:
        return FSDP_normal_trainer, FSDP_normal_validator, FSDP_normal_tester
    
    



def ddp_call_engines(args):
    if "hie" in args.model_name.lower():
        return DDP_hie_trainer, DDP_hie_validator, DDP_hie_tester
    else:
        return DDP_normal_trainer, DDP_normal_validator, DDP_normal_tester