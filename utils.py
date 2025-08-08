# class of GenerationIntervention conforming to spec laid out in obsidian
from collections import deque
import torch 
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoProcessor
import pickle
from typing import List, TypeVar, Dict, Tuple, Optional
import copy


class GeneralGenerationIntervention: # given a particular intervention, regenerate the downstream tokens and return
    
    def __init__(self, base_generation, causal_graph, model, tokenizer, acceptable):

        self.base_generation = base_generation # base generation as a string with placeholderss
        self.causal_graph = causal_graph

        self.model = model
        self.tokenizer = tokenizer

        self.acceptable = acceptable
        # manually set self.acceptable to ('int', [1, 9999] for all nodes)
        
        check_index_valid, bad_pairs = self.parents_before_children()
        print(bad_pairs)
        assert check_index_valid


    def _flatten_tokens(self, nested):
        """Yield every integer from an arbitrarily nested list/tuple structure."""
        for item in nested:
            if isinstance(item, (list, tuple)):
                yield from self._flatten_tokens(item)
            else:
                yield item

    def parents_before_children(self) -> Tuple[bool, List[Tuple[str, str]]]:
        """
        Return (is_ok, bad_pairs).

        *is_ok* is True only if, for every child in the causal graph, **each** of its parents
        appears **at least once** before the first occurrence of that child.

        *bad_pairs* contains every (parent, child) pair that violates the rule
        (including cases where either placeholder is missing from the template).
        """
        import re

        # 1. Build regex pattern to only match allowed placeholders
        allowed_vars = sorted(self.causal_graph.variables, key=len, reverse=True)
        escaped_vars = [re.escape(var) for var in allowed_vars]
        pattern = re.compile(r'\$(' + '|'.join(escaped_vars) + r')')

        # record all positions of each placeholder
        positions: Dict[str, List[int]] = {}
        for m in re.finditer(pattern, self.base_generation):
            var = m.group(1)
            positions.setdefault(var, []).append(m.start())

        # 2. Check parent/child relationships
        bad_pairs: List[Tuple[str, str]] = []

        for child, parents in self.causal_graph.parents.items():
            child_positions = positions.get(child)
            # if child never appears, every (parent, child) is automatically bad
            if not child_positions:
                bad_pairs.extend((p, child) for p in parents)
                continue

            first_child_pos = min(child_positions)

            for parent in parents:
                parent_positions = positions.get(parent)
                # if parent never appears, it's a violation
                if not parent_positions:
                    bad_pairs.append((parent, child))
                    continue

                # require at least one parent occurrence before the first child
                if not any(pos < first_child_pos for pos in parent_positions):
                    bad_pairs.append((parent, child))

        return (len(bad_pairs) == 0, bad_pairs)


    
    def _all_downstream(self, start_node: int):
        """Return *all* nodes reachable from `start_node` (excluding itself)."""
        visited, stack = set(), [start_node]
        while stack:
            node = stack.pop()
            for child in self.causal_graph.children.get(node, []):
                if child not in visited:
                    visited.add(child)
                    stack.append(child)
        visited.discard(start_node)
        return visited

    def _topo_sort(self, nodes):
        """Topologically sort the *induced* sub-graph on `nodes`."""
        indeg = {n: 0 for n in nodes}
        for u in nodes:
            for v in self.causal_graph.children.get(u, []):
                if v in nodes:
                    indeg[v] += 1

        q = deque([n for n, d in indeg.items() if d == 0])
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in self.causal_graph.children.get(u, []):
                if v in indeg:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        q.append(v)
        if len(order) != len(nodes):
            raise ValueError("Cycle detected in the downstream sub-graph.")
        return order

    def check_acceptable(self, expected_setting):
        

        for node in expected_setting:

            if node in self.acceptable:
                node_type, node_vals = self.acceptable[node]
                
                if node_type == 'int':

                    if expected_setting[node] < node_vals[0] or expected_setting[node] > node_vals[1]:
                        return False
                    else:
                        pass

        return True

    def generate_counterfactual_value(self, base_inputs, node, num_attempts=100):
        # generates counterfactual value for a given node, and checks if all predicted values under this counterfactual are in the acceptable range

        node_type, node_vals = self.acceptable[node]

        for _ in range(num_attempts):

            if node_type == 'int':

                new_setting = random.randint(self.acceptable[node][1][0], self.acceptable[node][1][1])

            intervened_setting = dict(base_inputs)
            intervened_setting[node] = new_setting
            expected_setting = dict(self.causal_graph.run_forward(intervened_setting))

            if self.check_acceptable(expected_setting):
                return new_setting
 
        return None

    def debug_forced_intervention(self, base_inputs, intervention_sites):

        site = 'S1'
        new_value = 93

        intervention_raw =  (site, new_value)
        intervention = (site, self.tokenizer.tokenize(str(new_value), add_special_tokens=False))

        intervened_setting = dict(base_inputs)
        intervened_setting[site] = new_value

        expected_setting = dict(self.causal_graph.run_forward(intervened_setting))

        expected_setting_raw = dict(expected_setting)
        expected_setting = {k:self.tokenizer.tokenize(str(expected_setting[k]), add_special_tokens=False) for k in expected_setting}

        return intervention, expected_setting, intervention_raw, expected_setting_raw
    
    def get_intervention(self, base_inputs, intervention_sites):

        # pick a random intervention site

        site = random.choice(intervention_sites)
        new_value = self.generate_counterfactual_value(base_inputs, site)

        if new_value is None:
            # raise an error
            raise ValueError(f"No counterfactual value found for site {site}.")

        intervention =  (site, new_value)

        # compute intervened list, assuming we always intervene on a partial sum
        intervened_setting = dict(base_inputs)
        intervened_setting[site] = new_value
        
        expected_setting = dict(self.causal_graph.run_forward(intervened_setting))
        expected_setting = dict(expected_setting)

        return intervention, expected_setting

    def intervened_generation(self, intervention, expected_setting):

        node_N, node_setting = intervention

        new_generation = copy.deepcopy(self.base_generation)

        downstream_nodes = self._all_downstream(node_N)
        order = self._topo_sort(downstream_nodes)

        for node in order:
            # 3a. Grab the expected tokens for this node.
            try:
                node_expected_setting = expected_setting[node]
            except IndexError:
                raise IndexError(f"Missing expected_setting for node {node}.")

            exp_tokens = self.tokenizer.tokenize(str(node_expected_setting), add_special_tokens=False)
            exp_len = len(exp_tokens)

            prefix_template = get_string_up_to_placeholder(self.base_generation, node)
            prefix = populate_placeholder_template(expected_setting, prefix_template)
            prefix = self.tokenizer(prefix, return_tensors="pt").to(self.model.device)

            generated  = self.model.generate(**prefix, max_new_tokens=exp_len, disable_compile=True, do_sample=False)
            new_tokens = self.tokenizer.convert_ids_to_tokens(generated[0, -exp_len:])

            # 3c. Check against expectation.
            if list(new_tokens) != list(exp_tokens):
                return False, (node, new_tokens), exp_tokens

        # ------------------------------------------------------------------
        # 4. Success – everything matched!
        # ------------------------------------------------------------------
        return True, new_generation, None

    def debug_intervened_generation(self, intervention, expected_setting):

        node_N, node_setting = intervention

        new_generation = copy.deepcopy(self.base_generation)

        downstream_nodes = self._all_downstream(node_N)
        order = self._topo_sort(downstream_nodes)
        print("Intervention")
        print(intervention)
        print("\n")
        print("expected setting")
        print(expected_setting)
        print("\n")

        for node in order:
            # 3a. Grab the expected tokens for this node.
            print("Currently generating node ", node)
            try:
                node_expected_setting = expected_setting[node]
            except IndexError:
                raise IndexError(f"Missing expected_setting for node {node}.")
            print("node expected setting")
            print(node_expected_setting)
            print("\n")

            exp_tokens = self.tokenizer.tokenize(str(node_expected_setting), add_special_tokens=False)
            print("exp tokens")
            print(exp_tokens)
            print("\n")
            exp_len = len(exp_tokens)
            print("exp len")
            print(exp_len)
            print("\n")
            prefix_template = get_string_up_to_placeholder(self.base_generation, node)
            print("prefix template")
            print(prefix_template)
            print("\n")
            prefix = populate_placeholder_template(expected_setting, prefix_template)
            print("prefix")
            print(prefix)
            print("\n")
            prefix_tokenized = self.tokenizer.tokenize(prefix, add_special_tokens=False)
            # 3b. Regenerate the *first* occurrence.
            prefix_ids = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(prefix_tokenized),
                dtype=torch.long,
                device=self.model.device
            ).unsqueeze(0)

            generated  = self.model.generate(prefix_ids, max_new_tokens=exp_len)
            new_tokens = self.tokenizer.convert_ids_to_tokens(generated[0, -exp_len:])

            print("generated")
            print(generated)
            print("\n")
            print("new tokens")
            print(new_tokens)
            print("\n")

            # 3c. Check against expectation.
            if list(new_tokens) != list(exp_tokens):
                return False, (node, new_tokens), exp_tokens

        # ------------------------------------------------------------------
        # 4. Success – everything matched!
        # ------------------------------------------------------------------
        return True, new_generation, None


# script utils

def load_model_and_tokenizer(model_name, load_in_8bit=True):
    print(f"Loading model {model_name} with load_in_8bit={load_in_8bit}")
    if model_name in ['gemma_3_1b_it', 'gemma_3_12b_it', 'gemma_3_27b_it']:
        tokenizer = AutoProcessor.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically distributes layers across GPUs
        cache_dir='/home/lfletcher/scratch',
        load_in_8bit=load_in_8bit
    )

    return model, tokenizer

def convert_model_name(shorthand):

    mapping = {
       "math_1_5":"Qwen/Qwen2.5-Math-1.5B",
       "math_7":"Qwen/Qwen2.5-Math-7B",
       "math_72":"Qwen/Qwen2.5-Math-72B",
       "math_1_5_instruct":"Qwen/Qwen2.5-Math-1.5B-Instruct",
       "math_7_instruct":"Qwen/Qwen2.5-Math-7B-Instruct",
       "math_72_instruct":"Qwen/Qwen2.5-Math-72B-Instruct",
       "llama_8":"meta-llama/Llama-3.1-8B-Instruct",
       "llama_70":"meta-llama/Llama-3.1-70B-Instruct",
       "gemma_3_1b_it":"google/gemma-3-1b-it",
       "gemma_3_4b_it":"google/gemma-3-4b-it",
       "gemma_3_12b_it":"google/gemma-3-12b-it",
       "gemma_3_27b_it":"google/gemma-3-27b-it"
    }

    return mapping.get(shorthand.lower(), "Unknown Model")

import re

def populate_placeholder_template(settings, template):
    # Regex pattern to find placeholders like $S1, $I1 etc.
    pattern = re.compile(r'\$([A-Za-z0-9_]+)')
    
    # Function to replace each placeholder
    def replace_match(match):
        key = match.group(1)
        return str(settings.get(key, match.group(0)))  # If key not found, leave placeholder as is
    
    # Replace all placeholders
    result = pattern.sub(replace_match, template)
    
    return result

def get_string_up_to_placeholder(template, node_N):
    placeholder = f"${node_N}"
    index = template.find(placeholder)
    
    if index == -1:
        raise ValueError(f"Placeholder {placeholder} not found in template.")
    else:
        return template[:index]


def get_intervention(base_setting, source_setting, intervention_sites, tokenizer):

    # pick a random intervention site

    site = random.choice(intervention_sites)

    intervention_raw =  (site, source_setting[site])
    intervention = (site, tokenizer.tokenize(str(source_setting[site]), add_special_tokens=False))

    # compute intervened list, assuming we always intervene on a partial sum

    expected_setting = list(base_setting)

    expected_setting[site] = source_setting[site]

    for n in range(site+1, len(expected_setting)):

        expected_setting[n] = expected_setting[n-1] + expected_setting[n - (len(base_setting) // 2)]

    expected_setting_raw = list(expected_setting)
    expected_setting = [tokenizer.tokenize(str(x), add_special_tokens=False) for x in expected_setting]

    return intervention, expected_setting, intervention_raw, expected_setting_raw


def get_instruct_model_inputs(prompt, tokenizer):

    if 'gemma' in tokenizer.name_or_path:
        print("Using gemma")
        messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Please reason step by step, and put your final answer within \\boxed{}."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
        ]

    else:
        print("Using non-gemma")
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": prompt}
        ]

    model_inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors='pt', return_dict=True).to('cuda')

    return model_inputs

def get_inputs(setting):
    """
    Extracts and returns a dictionary containing only keys of the form 'I' followed by digits.
    
    Parameters:
    setting (dict): Input dictionary with various keys.
    
    Returns:
    dict: Dictionary containing only 'I' keys with numeric suffixes.
    """
    return {k: v for k, v in setting.items() if k.startswith('I') and k[1:].isdigit()}

def get_intervention_sites(causal_graph):
    """
    returns variables which have both a parent and child (excludes inputs and final output)
    """
    return [key for key in causal_graph.parents if causal_graph.parents[key] and causal_graph.children[key]]


K = TypeVar('K')  # Generic type for keys
def get_node_token_indices_dict(
    tokens: List[str],
    patterns: Dict[K, List[str]],
    counts: Optional[Dict[K, int]] = None,
    *,
    allow_overlap: bool = True
) -> Dict[K, List[List[int]]]:
    """
    Return the indices of occurrences of each pattern (sublists) within tokens.
    Patterns are given as a dict: {key: [token, token, ...], ...}.
    Counts (if provided) is a dict {key: max_hits} limiting how many matches per pattern.

    Parameters
    ----------
    tokens : List[str]
        The master list of tokens to search through.
    patterns : Dict[K, List[str]]
        Mapping from arbitrary keys to the token‐sequence patterns to search for.
    counts : Optional[Dict[K, int]]
        How many occurrences to return for each pattern key.
        Must have exactly the same keys as `patterns` if provided.
        If None, all occurrences are returned.
    allow_overlap : bool, default True
        If True, overlapping matches are allowed; otherwise matches jump past each found block.

    Returns
    -------
    Dict[K, List[List[int]]]
        A dict mapping each key to a list of occurrences, where each occurrence
        is a list of the concrete indices in `tokens`.

    Example
    -------
    >>> tokens = ['a','b','c','a','b','c','a','b','c']
    >>> patterns = {'first': ['a','b'], 'second': ['b','c','a']}
    >>> counts = {'first': 2, 'second': 1}
    >>> get_node_token_indices(tokens, patterns, counts)
    {
      'first': [[0, 1], [3, 4]],
      'second': [[1, 2, 3]]
    }

    >>> get_node_token_indices(tokens, patterns)
    {
      'first': [[0, 1], [3, 4], [6, 7]],
      'second': [[1, 2, 3], [4, 5, 6]]
    }
    """
    if counts is not None and set(counts.keys()) != set(patterns.keys()):
        raise ValueError("If counts is provided, it must have exactly the same keys as patterns.")

    result: Dict[K, List[List[int]]] = {}

    for key, pattern in patterns.items():
        pattern_len = len(pattern)
        hits: List[List[int]] = []
        i = 0
        max_hits = counts[key] if counts is not None else float('inf')

        while i <= len(tokens) - pattern_len and len(hits) < max_hits:
            if tokens[i : i + pattern_len] == pattern:
                hits.append(list(range(i, i + pattern_len)))
                # advance by 1 or by full pattern length depending on overlap flag
                i += 1 if allow_overlap else pattern_len
            else:
                i += 1

        result[key] = hits

    return result