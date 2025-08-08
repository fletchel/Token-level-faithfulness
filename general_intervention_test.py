import argparse
from token_utils import *
import pandas as pd
import pickle
from causal_model import build_running_sum_model

def save_results(df, save_path):

    with open(save_path, 'wb') as f:

        pickle.dump(df, f)

def get_acceptable_counterfactuals_int(base_setting, acceptable_variations):

    acceptable_counterfactuals = {k: ('int', [max(1, base_setting[k] - acceptable_variations), min(9999, base_setting[k] + acceptable_variations)]) for k in base_setting.keys()}

    return acceptable_counterfactuals

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='math_1_5', help='which model to use (math_2, math_7, r1_2, r1_7)')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--num_interventions', type=int, default=100, help='number of interventions to perform')
    #parser.add_argument('--results_dir', type=str, default='../results/')
    parser.add_argument('--results_save_name', type=str, default='results')
    parser.add_argument('--acceptable_variations', type=int, default=30, help='amount +/- each side of base setting to consider acceptable for counterfactual generation')
    parser.add_argument('--load_in_8bit', action='store_true', default=False, help='whether to load the model in 8bit')

    args = parser.parse_args()

    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    full_model_name = convert_model_name(args.model)
    model, tokenizer = load_model_and_tokenizer(full_model_name, load_in_8bit=args.load_in_8bit)
    # generate dataset

    results_df = pd.DataFrame(columns=[
        'datapoint_idx',
        'intervention_idx',
        'new_tokens',
        'exp_tokens',
        'input_string',
        'causal_graph',
        'model_shorthand',
        'base_setting',
        'intervention',
        'intervention_raw',
        'expected_setting',
        'expected_setting_raw',
        'success',
        'error'
    ])

    num_successes = 0

    for i, datapoint in enumerate(data):

        print(f"Processing datapoint: {i}")

        if datapoint is None:
            cur_row = pd.DataFrame([{
                                'datapoint_idx':i,
                                'intervention_idx':0,
                                'new_tokens':None,
                                'exp_tokens':None,
                                'input_string':None,
                                'causal_graph':None,
                                'model_shorthand':args.model,
                                'base_setting':None,
                                'intervention':None,
                                'expected_setting':None,
                                'success':False,
                                'error':None}])

            results_df = pd.concat([results_df, cur_row], ignore_index=True)
            continue

        if 'input_string' not in datapoint:
            datapoint['input_string'] = datapoint['modified_string']
        
        input_string = datapoint['input_string']
        causal_graph = datapoint['causal_graph']
        acceptable = datapoint['acceptable_counterfactuals']
        base_setting = datapoint['base_setting']

        acceptable = get_acceptable_counterfactuals_int(base_setting, args.acceptable_variations)

        try:

            interv_obj = GeneralGenerationIntervention(input_string, causal_graph, model, tokenizer, acceptable)

        except Exception as e:

            error = str(e)
            print("Error at datapoint ", i, " during initialization")
            cur_row = pd.DataFrame([{
                                'datapoint_idx':i,
                                'intervention_idx':0,
                                'new_tokens':None,
                                'exp_tokens':None,
                                'input_string':input_string,
                                'causal_graph':causal_graph,
                                'model_shorthand':args.model,
                                 'base_setting':base_setting,
                                 'intervention':None,
                                 'expected_setting':None,
                                 'success':False,
                                 'error':error}])
            
            results_df = pd.concat([results_df, cur_row], ignore_index=True)
            continue

        num_cur_successes = 0

        for num_interv in range(args.num_interventions):
            print(f"Doing intervention {num_interv} for datapoint {i}")
            base_inputs = get_inputs(base_setting)
            intervention_sites = get_intervention_sites(causal_graph)
            try:
                intervention, expected_setting = interv_obj.get_intervention(base_inputs, intervention_sites)
                #intervention, expected_setting, intervention_raw, expected_setting_raw = interv_obj.debug_forced_intervention(base_inputs, intervention_sites)
                
                #success, new_tokens, exp_tokens = interv_obj.debug_intervened_generation(intervention, expected_setting)
                success, new_tokens, exp_tokens = interv_obj.intervened_generation(intervention, expected_setting)
            except Exception as e:
                print("Error at datapoint ", i, " intervention ", num_interv)
                print(e)
                error = str(e)
                success = False

                cur_row = pd.DataFrame([{
                                'datapoint_idx':i,
                                'intervention_idx':num_interv,
                                'new_tokens':None,
                                'exp_tokens':None,
                                'input_string':input_string,
                                'causal_graph':causal_graph,
                                'model_shorthand':args.model,
                                 'base_setting':base_setting,
                                 'intervention':None,
                                 'expected_setting':None,
                                 'success':False,
                                 'error':error}])

                results_df = pd.concat([results_df, cur_row], ignore_index=True)
                continue

            if success:
                
                new_tokens = None

            if success:
                num_cur_successes += 1
                num_successes += 1

            cur_row = pd.DataFrame([{
                                'datapoint_idx':i,
                                'intervention_idx':num_interv,
                                'new_tokens':new_tokens,
                                'exp_tokens':exp_tokens,
                                'input_string':input_string,
                                'causal_graph':causal_graph,
                                'model_shorthand':args.model,
                                 'base_setting':base_setting,
                                 'intervention':intervention,
                                 'expected_setting':expected_setting,
                                 'success':success,
                                 'error':None}])
        
            results_df = pd.concat([results_df, cur_row], ignore_index=True)

        print(f"Completed {args.num_interventions} interventions on datapoint {i}")
        print(f"Success proportion: {num_cur_successes/args.num_interventions}")
        print("\n")

    print("Completed all datapoints")
    print(f"Success rate: {num_successes/args.num_interventions}")
    
    save_path = args.results_save_name
    save_results(results_df, save_path)
        

    

if __name__ == "__main__":
    main()
