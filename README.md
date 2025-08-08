

### Prerequisites

Requirements are in the requirements.txt file

You'll also need to set your OpenAI API key as an environment variable:

    export OPENAI_API_KEY='your-api-key'

---
## How to Run the Scripts

The scripts in this repository are designed to be run in a specific order to generate data, process it, and run experiments.

### 1. Generate GSM8k Examples

This script generates text from a specified model based on questions from the GSM8k dataset.

    python generate_gsm_examples.py --model <model_name> --num_examples <number_of_examples> --results_dir <directory_to_save_results> --results_save_name <name_of_results_file>

**Arguments**:
* `--model`: The shorthand name of the model to use.
* `--num_examples`: The number of examples to generate.
* `--results_dir`: The directory to save the generated text.
* `--results_save_name`: The name of the file to save the results to.

### 2. Create Causal Graphs from API

This script takes the generated text and uses the OpenAI API to create causal graphs.

    python api_causal_graphs.py --data_path <path_to_generated_data> --model_short_name <model_shorthand> --indices_range <range> --output_path <path_to_save_output> --api_model <api_model_name>

**Arguments**:
* `--data_path`: The path to the file containing the generated text from the previous step.
* `--model_short_name`: The shorthand name of the model used for generation.
* `--indices_range`: The range of indices to process from the data file (e.g., "0-100").
* `--output_path`: The path to save the generated intervention prompts.
* `--api_model`: The API model to use for generating the causal graphs (e.g., "o4-mini").

### 3. Clean Up Causal Graphs

This script cleans the generated causal graphs by removing invalid or unsupported entries.

    python cleanup_api_causal_graphs.py --data_path <path_to_causal_graphs> --save_path <path_to_save_cleaned_data> --model_short_name <model_shorthand>

**Arguments**:
* `--data_path`: The path to the file containing the causal graphs from the previous step.
* `--save_path`: The path to save the cleaned data.
* `--model_short_name`: The shorthand for the model.

### 4. Run General Intervention Test

This is the main experimental script. It loads a model and the cleaned causal graphs, then performs interventions and records the results.

    python general_intervention_test.py --model <model_shorthand> --data_path <path_to_cleaned_data> --num_interventions <number_of_interventions> --results_save_name <name_for_results_file>

**Arguments**:
* `--model`: The shorthand name of the model to test.
* `--data_path`: The path to the cleaned intervention prompts.
* `--num_interventions`: The number of interventions to perform for each data point.
* `--results_save_name`: The name of the file to save the final results.


### Shorthand model names:

* **math_1_5**: "Qwen/Qwen2.5-Math-1.5B"
* **math_7**: "Qwen/Qwen2.5-Math-7B"
* **math_72**: "Qwen/Qwen2.5-Math-72B"
* **math_1_5_instruct**: "Qwen/Qwen2.5-Math-1.5B-Instruct"
* **math_7_instruct**: "Qwen/Qwen2.5-Math-7B-Instruct"
* **math_72_instruct**: "Qwen/Qwen2.5-Math-72B-Instruct"
* **llama_8**: "meta-llama/Llama-3.1-8B-Instruct"
* **llama_70**: "meta-llama/Llama-3.1-70B-Instruct"
* **gemma_3_1b_it**: "google/gemma-3-1b-it"
* **gemma_3_4b_it**: "google/gemma-3-4b-it"
* **gemma_3_12b_it**: "google/gemma-3-12b-it"
* **gemma_3_27b_it**: "google/gemma-3-27b-it"
