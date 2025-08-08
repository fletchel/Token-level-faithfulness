from pyvene import CausalModel
from causal_functions import add, subtract, multiply, zero_fn, divide, one_fn, copy
import json
from token_utils import get_node_token_indices_dict

def create_causal_model_from_spec(causal_graph_spec):

    variables = causal_graph_spec['variables']
    parents = causal_graph_spec['parents']
    values = {v: 1 for v in variables}
    functions = dict(causal_graph_spec['functions']) 

    for k in functions:

        if functions[k] == 'add':
            functions[k] = add

        elif functions[k] == 'subtract':
            functions[k] = subtract

        elif functions[k] == 'multiply':
            functions[k] = multiply
        
        elif functions[k] == 'divide':
            functions[k] = divide

        elif functions[k] == 'zero' or functions[k] == 'zero_fn':
            functions[k] = one_fn

        elif functions[k] in [add, subtract, multiply, divide, one_fn, copy]:
            pass
        
        else:
            # throw error
            raise ValueError(f"Function {functions[k]} not supported")

    return CausalModel(variables, values, parents, functions)





def api_call(messages, client, api_model):

    resp = client.chat.completions.create(
        model            = api_model,
        temperature      = 1,
        response_format  = {"type": "json_object"},
        messages=messages
    )

    return resp.choices[0].message.content

def parse_response(response):

    return json.loads(response)

def create_intervention_prompt(response):

    json = parse_response(response)

    if json['causal_graph_spec'] is None:
        datapoint = {'modified_string':None, 'causal_graph':None, 'acceptable_counterfactuals':None, 'base_setting':None}
        return datapoint
    
    causal_model = create_causal_model_from_spec(json['causal_graph_spec'])
    base_setting = json['base_setting']
    acceptable_counterfactuals = {v: ("int", [10, 9999]) for v in causal_model.variables}

    datapoint = {'modified_string':json['modified_string'], 'causal_graph':causal_model, 'acceptable_counterfactuals':acceptable_counterfactuals, 'base_setting':base_setting}

    return datapoint

def causal_graph_sanity_check(datapoint, tokenizer):

    print(tokenizer.convert_tokens_to_string(datapoint['tokenized_gen']))
    print(datapoint['causal_graph'].parents)
    print(datapoint['causal_graph'].functions)
    print(datapoint['base_setting'])
