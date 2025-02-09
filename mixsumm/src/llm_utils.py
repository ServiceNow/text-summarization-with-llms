import openai, time, numpy as np, os
from vllm import infer_vllm
from sentence_transformers import SentenceTransformer, util
from functools import partial

sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def ensure_extractiveness(summary, dialog_formatted):
    # post process summary with sbert
    split_summary = [s.strip() for s in summary.split(" [SEP] ") if s]
    split_dialogs = dialog_formatted.splitlines()
    encoded_dialogs = np.array([sbert.encode(l) for l in split_dialogs])
    encoded_summary_lines = np.array([sbert.encode(l) for l in split_summary])

    closest_line_ids = list(
        set(util.cos_sim(encoded_dialogs, encoded_summary_lines).argmax(-1).tolist())
    )
    new_summary = " [SEP] ".join([split_dialogs[i] for i in closest_line_ids])
    return new_summary


def mixsum(dialog_1, dialog_2, mode="naive", n_gen=1):
    if mode == "naive":
        return naive_mixsum(dialog_1, dialog_2, n_gen=n_gen)
    elif mode == "smart":
        return smart_mixsum(dialog_1, dialog_2, n_gen=n_gen)
    else:
        raise ValueError(f"Mode {mode} is not supported.")


def prompt_llm(system_desc, prompt):
    chat = partial(
        infer_vllm,
        max_tokens=10,
        end_point="llama-3-70b",
        top_logprobs=5,
        return_full_json=True,
    )
    try:
        response = chat(system_desc + "\n" + prompt)["choices"][0]["text"]
        return response
    except Exception as e:
        print(f"Error in llama scoring: {e}")
        return 0


def identify_issues_and_solutions(dialog_1, dialog_2):
    system_desc = "You are excellent agent that can precisely identify customer issues and agent solutions from a given conversation."
    instruction = "Given the conversation, identify the primary issue of the customer and the primary solution suggested by the agent. Generate your response in the format: 'Issue: <customer issue> [SEP] Solution: <agent solution>'."
    template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{conversation}

{instruction}


### Response:"""
    issues, solutions = [], []
    
    for dialog in [dialog_1, dialog_2]:
        prompt = template.format(
            conversation=f"Conversation:\n{dialog}",
            instruction=instruction,
        )

        while True:
            try:
                response = prompt_llm(system_desc, prompt)
                issue, solution = response.split(" [SEP] ")
                issues.append(issue.replace("Issue: ", "").strip())
                solutions.append(solution.replace("Solution: ", "").strip())
                break
            except openai.RateLimitError as e:
                print("RateLimitError, Sleeping for 100 seconds...")
                time.sleep(100)
            except openai.APIError as e:
                print(f"APIError, {e}\nSleeping for 100 seconds...")
                time.sleep(50)
            except Exception as e:
                print(f"{e}, Sleeping for 100 seconds...")
                time.sleep(50)
    return issues, solutions


def round_x(x):
    """Round a number to the nearest 0.05"""
    return round(x * 20) / 20


def sample_alpha_from_peaked_distribution(n_aug):
    """sample from a distribution between 0.5 and 1.0 with a peak near 1.0"""
    return [round_x(((x * 0.5) + 0.5)) for x in np.random.beta(5, 2, n_aug)]


def mix_issues_and_solutions(issues, solutions, alpha):
    """This function takes two lists containing issues and solutions, and asks LLM to mix them."""
    system_desc = "You are an expert in synthesizing customer agent conversations. You do that by mixing the issues and solutions from the two conversations to generate a new example."
    instruction = f"Generate a new issue that is {alpha}% like the first issue and {100-alpha}% like the second. Also generate a new solution that is {alpha}% like the first solution and {100-alpha}% like the second. Generate your response in the format: 'New issue: <customer issue> [SEP] New solution: <agent solution>'."
    template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{content}

{instruction}


### Response:"""
    content = (
        "Issue 1: "
        + issues[0]
        + "\nIssue 2: "
        + issues[1]
        + "\n\nSolution 1: "
        + solutions[0]
        + "\nSolution 2: "
        + solutions[1]
    )
    
    prompt = template.format(content=content, instruction=instruction)
    while True:
        try:
            response = prompt_llm(system_desc, prompt)
            issue, solution = response.split(" [SEP] ")
            return (
                issue.replace("New issue: ", "").strip(),
                solution.replace("New solution: ", "").strip(),
            )
        except openai.RateLimitError as e:
            print("RateLimitError, Sleeping for 100 seconds...")
            time.sleep(100)
        except openai.APIError as e:
            print(f"APIError, {e}\nSleeping for 100 seconds...")
            time.sleep(50)
        except Exception as e:
            print(f"{e}, Sleeping for 100 seconds...")
            time.sleep(50)


def synthesize_new_dialog(issue, solution, example_dialog=None):
    """This function takes an issue and a solution and asks LLM to synthesize a new dialog."""
    
    system_desc = "You are an expert in synthesizing customer agent conversations. You do that by generating a new conversation based on the issue and solution provided."
    instruction = "Generate a new conversation that includes the new issue and solution above. Your response should be formatted like the included example conversation and should be similar in length."
    template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{content}

{instruction}

### Response:
New conversation:"""
    prompt = template.format(
        content=f"Example Conversation: {example_dialog}\n\nIssue: {issue}\nSolution: {solution}",
        instruction=instruction,
    )
    
    while True:
        try:
            response = prompt_llm(system_desc, prompt)
            return response.replace("\n\n", "\n")
        except openai.RateLimitError as e:
            print("RateLimitError, Sleeping for 100 seconds...")
            time.sleep(100)
        except openai.APIError as e:
            print(f"APIError, {e}\nSleeping for 50 seconds...")
            time.sleep(50)
        except Exception as e:
            print(f"{e}, Sleeping for 50 seconds...")
            time.sleep(50)


def smart_mixsum(dialog_1, dialog_2, n_gen=1):
    """This function takes two dialogs and returns a mix of the two dialogs."""
    
    dialog_1 = dialog_1.replace(" [SEP] ", "\n")
    dialog_2 = dialog_2.replace(" [SEP] ", "\n")
    issues, solutions = identify_issues_and_solutions(dialog_1, dialog_2)
    alphas = sample_alpha_from_peaked_distribution(n_gen)
    new_summaries = []
    for alpha in alphas:
        new_issue, new_solution = mix_issues_and_solutions(issues, solutions, alpha)
        new_summaries.append(synthesize_new_dialog(new_issue, new_solution, dialog_1))
    return new_summaries


def naive_mixsum(dialog_1, dialog_2, n_gen=1):
    """This function takes two dialogs and returns a mix of the two dialogs."""
    
    dialog_1 = dialog_1.replace(" [SEP] ", "\n")
    dialog_2 = dialog_2.replace(" [SEP] ", "\n")
    system_desc = "You are an expert in synthesizing customer agent conversations. You do that identifying the type of customer issues and agent solutions from the two conversations, and mixing them to generate a new example."
    template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{conversation}

{instruction}


### Response:
New conversation:"""
    prompt = template.format(
        conversation=f"Dialog #1:\n{dialog_1}\n\nDialog #2:\n{dialog_2}",
        instruction=system_desc,
    )

    
    summaries = []
    for _ in range(n_gen):
        while True:
            try:
                response = prompt_llm(system_desc, prompt)
                summaries.append(response.replace("\n\n", "\n"))
                break
            except openai.RateLimitError as e:
                print("RateLimitError, Sleeping for 100 seconds...")
                time.sleep(100)
            except openai.APIError as e:
                print(f"APIError, {e}\nSleeping for 50 seconds...")
                time.sleep(50)
            except Exception as e:
                print(f"{e}, Sleeping for 50 seconds...")
                time.sleep(50)
    return summaries


def no_mixup(dialog_1, n_gen=1):
    """This function takes two dialogs and returns a mix of the two dialogs."""
    
    dialog_1 = dialog_1.replace(" [SEP] ", "\n")
    system_desc = "You are an expert in synthesizing customer agent conversations. You do that by analyzing the customer issues and agent solutions from the conversation and generating a new example."
    template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{conversation}

{instruction}


### Response:
New conversation:"""
    prompt = template.format(
        conversation=f"Dialog #1:\n{dialog_1}\n\n",
        instruction=system_desc,
    )

    
    summaries = []
    for _ in range(n_gen):
        while True:
            try:
                response = prompt_llm(system_desc, prompt)
                summaries.append(response.replace("\n\n", "\n"))
                break
            except openai.RateLimitError as e:
                print("RateLimitError, Sleeping for 100 seconds...")
                time.sleep(100)
            except openai.APIError as e:
                print(f"APIError, {e}\nSleeping for 50 seconds...")
                time.sleep(50)
            except Exception as e:
                print(f"{e}, Sleeping for 50 seconds...")
                time.sleep(50)
    return summaries


def llm_relabelling(dialog):
    """This function asks LLM to output the probability of occurance of every sentence in the dialog and constructs an extractive summary using those values."""
    
    dialog_split = dialog.splitlines()
    system_desc = "For each sentence, output the probability of it appearing in the summary of an article made by joining all the lines together in sequence. The summary captures the main points of the article. Output in the format <line id>. <probability>."
    dialog_formatted = "\n".join([f"{i+1}. {l}" for i, l in enumerate(dialog_split)])
    template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{conversation}

{instruction}

### Response:"""
    prompt = template.format(
        # content="Provide probability of a sentence appearing a summary constructed by concatenating the lines shown. The summary satisfies the following criteria:\n1) The summary should be concise, ideally with less than four sentences.\n2) The summary should capture the main issue of the customer and the main solution suggested by the agent. \n3) The summary should be an extractive summary, directly taken from the conversation without any modifications or removal of symbols.\n\nCheck very carefully.",
        conversation=f"{dialog_formatted}",
        instruction=system_desc,
    )

    chat = partial(
        infer_vllm,
        max_tokens=10,
        end_point="llama-3-70b",
        top_logprobs=5,
        return_full_json=True,
    )
    while True:
        try:
            response = chat(prompt)["choices"][0]["text"]
            try:
                scores = [
                    float(r.split(".", 1)[1].replace("%", "").strip())
                    for r in response.splitlines()
                ]
                sent_scores = list(zip(range(len(dialog_split)), scores, dialog_split))
                sent_scores = sorted(sent_scores, key=lambda o: o[1], reverse=True)[:4]
                sent_scores = sorted(sent_scores, key=lambda o: o[0])
                summary = " [SEP] ".join([o[2] for o in sent_scores])
            except Exception as e:
                print("Error when parsing LLM probabilities: ", e)
                summary = ""
            if summary:
                return summary
        except openai.RateLimitError as e:
            print("RateLimitError, Sleeping for 100 seconds...")
            time.sleep(100)
        except openai.APIError as e:
            print(f"APIError, {e}\nSleeping for 100 seconds...")
            time.sleep(100)
        except Exception as e:
            print(f"{e}, Sleeping for 100 seconds...")
            time.sleep(100)
