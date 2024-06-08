import string
import os
import csv
import timeit
import json
from vllm import LLM, SamplingParams


def parse_mmlu_response(mmlu_example, model_output):
    model_output = model_output.lower()
    model_output = model_output.translate(str.maketrans('', '', string.punctuation))
    output_words = model_output.strip().split()
    check = output_words[0:4] == ['the', 'correct', 'answer', 'is']
    if (check and output_words[4] in ['a', 'b', 'c', 'd']):
        return output_words[4].upper()
    return None


def evaluate_performance_mmlu(dir_path, model_name):
    prompts, correct_answers = get_mmlu_prompts(dir_path)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024, stop=["\n"])
    if model_name == 'base':
        llm = LLM(model='/data/Meta-Llama-3-8B')
    elif model_name == 'instruct':
        llm = LLM(model='/home/shared/Meta-Llama-3-70B-Instruct')
    
    start_time = timeit.default_timer()
    outputs = llm.generate(prompts, sampling_params)
    end_time = timeit.default_timer()
    throughput = len(outputs) / (end_time - start_time)
    incorrect_predictions = []
    failed_parse = []

    num_fail_parse = 0
    num_correct = 0
    for output in outputs:
        prompt = output.prompt
        prompt_index = prompts.index(prompt)
        generated_text = output.outputs[0].text
        answer = parse_mmlu_response(prompt, generated_text)
        if answer == None:
            num_fail_parse += 1
            failed_parse.append((prompt, generated_text))
        elif answer == correct_answers[prompt_index]:
            num_correct += 1
        else:
            incorrect_predictions.append((prompt, generated_text, answer, correct_answers[prompt_index]))
    
    accuracy = num_correct / len(outputs)
    json_output = {
        "accuracy": accuracy,
        "throughput": throughput,
        "num_fail_parse": num_fail_parse,
        "incorrect_predictions": incorrect_predictions,
        "failed_parse": failed_parse
    }
    with open('mmlu_eval.json', 'w') as f:
        json.dump(json_output, f, indent=4)

def get_mmlu_prompts(dir_path):
    prompts = []
    correct_answers = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(dir_path, filename)
            subject = filename[:-4]
            with open(filepath, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    question = row[0]
                    options = row[1:5]
                    actual_answer = row[5]
                    
                    instruction = (f"Answer the following multiple choice question about {subject}. Respond with a single sentence of the "
                                   f"form \"The correct answer is _\", filling the blank with the letter corresponding to the correct answer "
                                   f"(i.e., A, B, C or D).\nQuestion: {question}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nAnswer")
                    
                    prompt = (f"# Instruction\nBelow is a list of conversations between a human and an AI assistant (you).\n"
                              f"Users place their queries under \"# Query:\", and your responses are under \"# Answer:\".\n"
                              f"You are a helpful, respectful, and honest assistant.\n"
                              f"You should always answer as helpfully as possible while ensuring safety.\n"
                              f"Your answers should be well-structured and provide detailed information. They should also have an engaging tone.\n"
                              f"Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic,dangerous, or illegal content, even if it may be helpful.\n"
                              f"Your response must be socially responsible, and thus you can reject to answer some controversial topics.\n"
                              f"# Query:\n"
                              f"```{instruction}```\n"
                              f"# Answer:\n"
                              f"```")
                    prompts.append(prompt)
                    correct_answers.append(actual_answer)
    return prompts, correct_answers


def main():
    dir_path = './data/mmlu/test'
    model_name = 'base'
    evaluate_performance_mmlu(dir_path, model_name)


if __name__ == "__main__":
    main()