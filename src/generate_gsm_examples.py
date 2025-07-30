import argparse
from token_utils import load_model_and_tokenizer, convert_model_name, get_instruct_model_inputs
from datasets import load_dataset
import random
import pickle
import torch

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='math_7_instruct', help='which model to use (math_2, math_7, r1_2, r1_7)')
    parser.add_argument('--num_examples', type=int, default=100, help='size of dataset to generate')
    parser.add_argument('--results_dir', type=str, default='../gsm_gens/')
    parser.add_argument('--results_save_name', type=str, default='gens')
    parser.add_argument('--load_in_8bit', action='store_true', help='load the model in 8bit')
    #parser.add_argument('--indices_range', type=str, default='0-100', help='range of indices to generate examples for')

    args = parser.parse_args()

    ds = load_dataset("openai/gsm8k", "main")

    model, tokenizer = load_model_and_tokenizer(convert_model_name(args.model), load_in_8bit=args.load_in_8bit)
    # select n datapoints from the test set
    #datapoint_indices = random.sample(list(range(len(ds['test']['question']))), args.num_examples)
    #datapoint_indices = list(range(int(args.indices_range.split('-')[0]), int(args.indices_range.split('-')[1])))
    datapoint_indices = list(range(200, 500))
    #datapoint_indices = list(range(args.num_examples))

    gens = []
    answers = []

    for i in datapoint_indices:

        print(f"Generating example {i} of {args.num_examples}")

        cur_prompt = ds['test']['question'][i]
        cur_answer = ds['test']['answer'][i]
        
        model_input = get_instruct_model_inputs(cur_prompt, tokenizer)
        print(model_input)
        gen_tokens = model.generate(**model_input, max_new_tokens=1000)
        gens.append(tokenizer.decode(gen_tokens[0]))
        print(gens[-1])
        answers.append(cur_answer)


    results = {'gens':gens, 'answers':answers}

    save_path = args.results_dir + args.results_save_name + '.pkl'

    with open(save_path, 'wb') as f:

        pickle.dump(results, f)

if __name__ == '__main__':
    main()