'''
We need to clean up the api generated causal graphs by doing the following:

- remove any nones
- remove any containing a division
- remove any which don't follow the I, S, O format
- remove any where any tokenized value in the base setting is a sublist of another tokenized value in the base setting
'''

import argparse
import pickle
from token_utils import convert_model_name
from transformers import AutoTokenizer
import re
from typing import List
from causal_functions import divide

def append_none(data):
    data.append(None)

def normalize_dollar_vars(text: str, vars_list: List[str]) -> str:
    """
    If text contains at least one occurrence of a dollar-prefixed variable (e.g. "$I1")
    AND at least one un-prefixed occurrence of a variable in vars_list (e.g. "I2"),
    then prepend a '$' to every occurrence of every var in vars_list.
    
    Example:
      vars_list = ["I1", "I2"]
      text = "We have $I1 and I2."
      → "We have $$I1 and $I2."
    """
    # Build an alternation like "I1|I2|S1|O"
    var_pattern = "|".join(re.escape(v) for v in vars_list)
    
    # Matches exactly one '$' before the var, but not two: (?<!\$)\$(I1|I2)\b
    prefixed_re = re.compile(r'(?<!\$)\$(' + var_pattern + r')\b')
    # Matches the var without any '$' immediately before it: (?<!\$)\b(I1|I2)\b
    unprefixed_re = re.compile(r'(?<!\$)\b('    + var_pattern + r')\b')
    
    # Do we have at least one of each?
    if prefixed_re.search(text) and unprefixed_re.search(text):
        # 1) Double-up the already-prefixed ones: "$I1" → "$$I1"
        text = prefixed_re.sub(lambda m: '$$' + m.group(1), text)
        # 2) Prefix the previously-unprefixed ones: "I2" → "$I2"
        text = unprefixed_re.sub(lambda m: '$'  + m.group(1), text)
    
    return text

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='../data/intervention_prompts.pkl', help='path to the data file')
    parser.add_argument('--save_path', type=str, default='../data/intervention_prompts_cleaned.pkl', help='path to save the cleaned data')
    parser.add_argument('--model_short_name', type=str, default='math_7_instruct', help='short name of the model')

    args = parser.parse_args()

    model_name = convert_model_name(args.model_short_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(args.data_path, 'rb') as f:

        data = pickle.load(f)


    cleaned_data = []
    failed_number = 0

    for i in range(len(data)):


        if data[i] is None:
            append_none(cleaned_data)
            continue

        if 'input_string' in data[i]:
            data[i]['modified_string'] = data[i]['input_string']

        if data[i]['causal_graph'] is None:
            append_none(cleaned_data)
            continue

        if 'division' in data[i]['causal_graph'].functions.values():
            append_none(cleaned_data)
            continue

        if divide in data[i]['causal_graph'].functions.values():
            append_none(cleaned_data)
            continue
        
        data[i]['modified_string'] = normalize_dollar_vars(data[i]['modified_string'], data[i]['causal_graph'].variables)

        cleaned_data.append(data[i])

    print(f"Number of datapoints: {len(cleaned_data)}")
    print(f"Number of nones: {cleaned_data.count(None)}")

    with open(args.save_path, 'wb') as f:

        pickle.dump(cleaned_data, f)

