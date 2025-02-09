import time, numpy as np, openai, re, os
import prompts
from functools import partial
from vllm import infer_vllm
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


def llm_relabelling(dialog):
    """This function asks LLM to output the probability of occurance of every sentence in the dialog and constructs an extractive summary using those values."""
    return llama_relabelling(dialog)


def get_llm_score(summary, curr_dialog, top_logprobs=None):
    return get_llama_score(summary, curr_dialog, top_logprobs)


def get_llama_score(summary, curr_dialog, top_logprobs=None):
    chat = partial(
        infer_vllm,
        max_tokens=10,
        end_point="llama-3-70b",
        top_logprobs=5,
        return_full_json=True,
    )
    human_message = prompts.LLAMA_SCORE_TEMPLATE
    if len(curr_dialog.splitlines()) > 1:
        dialog_split = curr_dialog.splitlines()
    else:
        dialog_split = curr_dialog.split(" . ")
    curr_dialog = ""
    for i, l in enumerate(dialog_split):
        if len(curr_dialog.split()) < 4000:
            curr_dialog += f"{l} . "

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template(human_message),
        ]
    )

    formatted_messages = chat_prompt.format_prompt(
        gt_summary=curr_dialog, summary=summary
    ).to_messages()

    try:
        formatted_messages = convert_messages_to_text(formatted_messages)
        response = chat(formatted_messages)

        rating_str = re.findall(
            r"<rating>(\d+)</rating>", response["choices"][0]["text"]
        )[0]
        rating_idx_in_response = response["choices"][0]["logprobs"]["tokens"].index(
            rating_str
        )
        response = response["choices"][0]["logprobs"]["top_logprobs"][
            rating_idx_in_response
        ]

        # convert logprobs to probs
        probs = [np.exp(logprob) for token, logprob in response.items()]
        # renormalize probs to sum to 1
        probs = [obj / sum(probs) for obj in probs]
        ratings = [
            float(token) if token.isdigit() else 0 for token, _ in response.items()
        ]
        # final score
        score = sum([a * b for a, b in zip(ratings, probs)])
        score = float(score)
        return score
    except Exception as e:
        print(f"Error in llama scoring: {e}")
        return 0


def llama_relabelling(dialog):
    chat = partial(
        infer_vllm,
        max_tokens=1000,
        end_point="llama-3-70b",
        top_logprobs=None,
        return_full_json=False,
    )

    dialog_split = (
        dialog.splitlines() if len(dialog.splitlines()) > 1 else dialog.split(" . ")
    )  # split by lines or sentences
    dialog_formatted = ""
    for i, l in enumerate(dialog_split):
        # adjust dialog length to fit model context length
        if len(dialog_formatted.split()) < 4000:
            dialog_formatted += f"{i+1}. {l}\n"
    human_message = prompts.LLM_RELABELLING_TEMPLATE

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template(human_message),
        ]
    )

    formatted_messages = chat_prompt.format_prompt(
        sentences=dialog_formatted
    ).to_messages()

    try:
        formatted_messages = convert_messages_to_text(formatted_messages)
        # print(f"Formatted messages: {formatted_messages}")
        response = chat(formatted_messages)
        # print(f"Response: {scores}")
        scores = re.findall(r"\d+\.\s([01].\d+)", response)
        sent_scores = list(zip(range(len(scores)), scores, dialog_split))
        sent_scores = sorted(sent_scores, key=lambda o: o[1], reverse=True)[:8]
        sent_scores = sorted(sent_scores, key=lambda o: o[0])
        limiter = "[SEP]" if len(dialog.splitlines()) > 1 else "."
        summary = f" {limiter} ".join([o[2] for o in sent_scores])
        return summary
    except Exception as e:
        print(f"Error in llama relabelling: {e}")
        return ""


def convert_messages_to_text(messages):
    return "\n".join(
        [
            (
                f"[INST]\n{m.content}\n[/INST]"
                if m.type in ["system", "agent"]
                else f"\n{m.content}\n"
            )
            for m in messages
        ]
    )