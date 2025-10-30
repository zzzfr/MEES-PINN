from flax.core.frozen_dict import FrozenDict, unfreeze

def process_variables(variables):
    if isinstance(variables, FrozenDict):
        variables = unfreeze(variables)

    while isinstance(variables, dict) and "params" in variables:
        inner = variables["params"]
        if isinstance(inner, dict):
            variables = inner
        else:
            break

    return {"params": variables}
