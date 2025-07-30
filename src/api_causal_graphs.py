import openai
import pickle
from api_tools import create_causal_model_from_spec, create_intervention_prompt, api_call
from prompts import create_messages
from transformers import AutoTokenizer
from token_utils import convert_model_name
import os 
from argparse import ArgumentParser

key = os.getenv('OPENAI_API_KEY')

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../gsm_gens/gsm_gens_7.pkl")
    parser.add_argument("--model_short_name", type=str, default="math_7_instruct")
    parser.add_argument("--indices_range", type=str, default="0-100")
    parser.add_argument("--output_path", type=str, default="../data/intervention_prompts.pkl")
    parser.add_argument("--api_model", type=str, default="o4-mini")
    parser.add_argument("--nodes", action="store_true")
    args = parser.parse_args()

    model_name = convert_model_name(args.model_short_name)
    client = openai.OpenAI(api_key=key)
    
    with open(args.data_path, 'rb') as f:

        data = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = []
    intervention_prompts = []

    for i in range(int(args.indices_range.split('-')[0]), int(args.indices_range.split('-')[1])):

        cur_gen = data['gens'][i]

        messages.append((create_messages(cur_gen), cur_gen))


    print("Completed messages")
    
    for i in range(len(messages)):

        print(f"Processing {i} of {len(messages)}")
        try:
            response = api_call(messages[i][0], client, args.api_model)
            intervention_prompts.append(create_intervention_prompt(response))
        except Exception as e:
            print(e)
            print(f"Error processing {i}")
            intervention_prompts.append(None)
            continue

    print("Completed intervention prompts")
    
    with open(args.output_path, 'wb') as f:

        pickle.dump(intervention_prompts, f)
