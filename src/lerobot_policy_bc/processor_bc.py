# processor_bc.py
from typing import Dict, Any
from lerobot.policies.processing import PolicyProcessorPipeline, PolicyAction

def make_bc_pre_post_processors(config):
    preprocessor = None # lerobot handles standard preprocessing
    # Postprocess - select n_action_steps from predicted horizon
    def postprocess(action: PolicyAction) -> PolicyAction:
        # All horizon steps predicted; select first n_action_steps
        action.values = action.values[:config.n_action_steps]
        return action
    
    postprocessor = PolicyProcessorPipeline([postprocess])

    return preprocessor, postprocessor