import pickle
from copy import deepcopy

with open('../gsm_gens/gsm_test_gens.pkl', 'rb') as f:
    gsm_gens = pickle.load(f)

for g in gsm_gens['gens']:

    if 'good seashells' in g:
        sea_gen = g

with open("/home/lfletcher/wc_23_jun_exp/gsm_gens/gsm_gens_7.pkl", "rb") as f:

    gens = pickle.load(f)

for i, g in enumerate(gens['gens']):
    if 'Charlie has left' in g:
        sticker_gen = g

example_input_no = gsm_gens['gens'][0]

example_input_yes_1 = sea_gen
example_input_yes_2 = sticker_gen

base_setting_yes_1 = {
    "I1": 214,
    "I2": 432,
    "I3": 86,
    "I4": 67,
    "S1": 646,
    "S2": 732,
    "O": 665
  }

base_setting_yes_2 = {'I1': 10, 'I2': 21, 'I3': 23, 'I4': 9, 'I5': 28, 'S1': 31, 'S2': 54, 'S3': 45, 'O': 17}

variable_gen_yes_1 = deepcopy(example_input_yes_1)
variable_gen_yes_1 = variable_gen_yes_1.replace(str(base_setting_yes_1["I1"]), "$I1")
variable_gen_yes_1 = variable_gen_yes_1.replace(str(base_setting_yes_1["I2"]), "$I2")
variable_gen_yes_1 = variable_gen_yes_1.replace(str(base_setting_yes_1["I3"]), "$I3")
variable_gen_yes_1 = variable_gen_yes_1.replace(str(base_setting_yes_1["I4"]), "$I4")
variable_gen_yes_1 = variable_gen_yes_1.replace(str(base_setting_yes_1["S1"]), "$S1")
variable_gen_yes_1 = variable_gen_yes_1.replace(str(base_setting_yes_1["S2"]), "$S2")
variable_gen_yes_1 = variable_gen_yes_1.replace(str(base_setting_yes_1["O"]), "$O")

variable_gen_yes_2 = deepcopy(example_input_yes_2)
variable_gen_yes_2 = variable_gen_yes_2.replace(str(base_setting_yes_2["I1"]), "$I1")
variable_gen_yes_2 = variable_gen_yes_2.replace(str(base_setting_yes_2["I2"]), "$I2")
variable_gen_yes_2 = variable_gen_yes_2.replace(str(base_setting_yes_2["I3"]), "$I3")
variable_gen_yes_2 = variable_gen_yes_2.replace(str(base_setting_yes_2["I4"]), "$I4")
variable_gen_yes_2 = variable_gen_yes_2.replace(str(base_setting_yes_2["I5"]), "$I5")
variable_gen_yes_2 = variable_gen_yes_2.replace(str(base_setting_yes_2["S1"]), "$S1")
variable_gen_yes_2 = variable_gen_yes_2.replace(str(base_setting_yes_2["S2"]), "$S2")
variable_gen_yes_2 = variable_gen_yes_2.replace(str(base_setting_yes_2["S3"]), "$S3")
variable_gen_yes_2 = variable_gen_yes_2.replace(str(base_setting_yes_2["O"]), "$O")

# have printed and checked these make sense

SYSTEM_MSG = """You are a strict JSON emitter. 
When the user gives you an input string, produce a JSON object
whose keys and value types follow the specification they give you exactly. 
Return ONLY valid JSON – no comments, code fences, or prose.
"""

# removed "if you are unsure, return None for all the keys"

FIRST_PROMPT_TEMPLATE = """
Below is a Python string produced by a GPT model while it solved a
simple maths word-problem.  You have two jobs

Firstly, you must check that the input satisfies the following conditions:
1) All variables must be expressed as integer numerals in the text. For instance, they must not be written as words, or have a percentage sign attached. For example, "ten", "twice" or "0.1" is not valid, while "10" or "2" is. You should take all instances of a number into account when evaluating this condition. For instance, it may be that in the model's reasoning, it refers to a variable correctly as "2", but in the input, it was referred to as "two". In this case, you MUST return None.
2) The causal graph must only involve functions we have defined.
3) The chain of thought must follow a clear algorithm. No mistakes should be made.

If the input does not satisfy ANY of these conditions, return None for all the keys.

If the input satisfies these conditions, you must then reconstruct the implied causal graph as follows:

1.  Reconstruct the implied causal graph.  Each variable name must be a valid
    Python identifier.  Use inputs ["I1","I2",…], intermediates ["S1","S2",…],
    and single output "O". Intermediates must ALWAYS be named S1, S2 etc.

2.  Provide the necessary information for constructing the causal graph, as follows:
       - variables  = inputs + intermediates + ["O"]
       - parents    = {{ as required }}
       - functions  = {{ zero_fn for inputs, add / subtract / multiply for others }}
       - values     = {{v:0 for v in variables}}

3.  Return the input string with all occurrences of a variable replaced with a placeholder. For instance, you might replace "he had 123 lemons" with "he had $I1 lemons". For all other tokens, return them EXACTLY as they appear. It is extremely important that other tokens are not replaced.

4.  Provide `base_setting`: the integer value of each variable as it appears
    in the worked solution (read it off the text).

**Return exactly one JSON object** whose keys are: causal_graph_spec, base_setting, modified_string

Note: The inputs must *ALWAYS* be in the user question. If a variable appears in the assistant response, but not in the user input, and it is not an intermediate variable, do not include it.

Note: You must return the provided string EXACTLY AS IS, other than replacing the variables with placeholders.

If it is not possible to construct a causal graph in this format from the input string, return None for all the keys. 
You should always return None for all the keys if
1) The inputs are not expressed as integer numerals. For instance, if they are written as words, or if they have a percentage sign attached.
2) The causal graph involves functions we haven't defined.

STRING BEGINS HERE:

{input_string_here}
"""

EXAMPLE_USER_YES_PLACEHOLDERS_1 = example_input_yes_1
EXAMPLE_ASSISTANT_JSON_YES_PLACEHOLDERS_1 = { 
      "causal_graph_spec": {
    "inputs": ["I1", "I2", "I3", "I4"],
    "intermediate": ["S1", "S2"],
    "variables": ["I1", "I2", "I3", "I4", "S1", "S2", "O"],
    "parents": {
      "I1": [],
      "I2": [],
      "I3": [],
      "I4": [],
      "S1": ["I1", "I2"],
      "S2": ["S1", "I3"],
      "O":  ["S2", "I4"]
    },
    "functions": {
      "I1": "zero",
      "I2": "zero",
      "I3": "zero",
      "I4": "zero",
      "S1": "add",
      "S2": "add",
      "O":  "subtract"
    }
  },

  "base_setting": {
    "I1": 214,
    "I2": 432,
    "I3": 86,
    "I4": 67,
    "S1": 646,
    "S2": 732,
    "O": 665
  },

  "modified_string": variable_gen_yes_1}

EXAMPLE_USER_YES_PLACEHOLDERS_2 = example_input_yes_2
EXAMPLE_ASSISTANT_JSON_YES_PLACEHOLDERS_2 = { 
    "causal_graph_spec": {
        "inputs": ["I1", "I2", "I3", "I4", "I5"],
        "intermediate": ["S1", "S2", "S3"],
        "variables": ["I1", "I2", "I3", "I4", "I5", "S1", "S2", "S3", "O"],
        "parents": {
            "I1": [],
            "I2": [],
            "I3": [],
            "I4": [],
            "I5": [],
            "S1": ["I1", "I2"],
            "S2": ["S1", "I3"],
            "S3": ["S2", "I4"],
            "O":  ["S3", "I5"]
        },
        "functions": {
            "I1": "one",
            "I2": "one",
            "I3": "one",
            "I4": "one",
            "I5": "one",
            "S1": "add",
            "S2": "add",
            "S3": "subtract",
            "O":  "subtract"
        }
    },

    "base_setting": base_setting_yes_2,

    "modified_string": variable_gen_yes_2
}


EXAMPLE_USER_NO = example_input_no
EXAMPLE_ASSISTANT_JSON_NO = {
  "modified_string": None,

  "causal_graph_spec": None,

  "base_setting": None
}

STRING_BEGINS_HERE = """STRING BEGINS HERE:

{input_string_here}"""

example_yes_1 = FIRST_PROMPT_TEMPLATE.format(input_string_here=example_input_yes_1)
example_yes_2 = STRING_BEGINS_HERE.format(input_string_here=example_input_yes_2)
example_no = STRING_BEGINS_HERE.format(input_string_here=example_input_no)

def create_messages(tokenized_gen, nodes=False):

    REAL_USER_PROMPT = STRING_BEGINS_HERE.format(input_string_here=tokenized_gen)

    if not nodes:
        messages = [
        # 0.  Guard-rail / “only JSON” rule
        {"role": "system", "content": SYSTEM_MSG},
        
        # 1.  One-shot EXAMPLE ── user gives a short token list
        {"role": "user", "content": example_yes_1},
        # 2.  One-shot EXAMPLE ── assistant shows the expected JSON
        {"role": "assistant", "content": str(EXAMPLE_ASSISTANT_JSON_YES_PLACEHOLDERS_1)},

        {"role": "user", "content": example_yes_2},
        # 2.  One-shot EXAMPLE ── assistant shows the expected JSON
        {"role": "assistant", "content": str(EXAMPLE_ASSISTANT_JSON_YES_PLACEHOLDERS_2)},

        {"role": "user", "content": example_no},

        {"role": "assistant", "content": str(EXAMPLE_ASSISTANT_JSON_NO)},
        # 3.  Your real user query
        {"role": "user", "content": REAL_USER_PROMPT}
        ]

    return messages
