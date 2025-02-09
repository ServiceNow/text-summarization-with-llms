LLM_RELABELLING_TEMPLATE = """
### Instruction:
A document is composed of the following sentences:
{sentences}

* You are to construct an extractive summary for the document by selecting some sentences from above.
* The summary captures the main points of the article.
* Now, output the probability of a sentence being included in the summary.
* Do NOT include anything else like the sentence in your output.
* Output your probabilities in the format <line id>. <probability>. Refer to the example below:
1. 0.73
2. 0.65
3. 0.95
etc.

### Response:
"""

LLM_SCORE_TEMPLATE = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Given the article:
{article}

And given the summary:
{summary}

Provide a rating between 1-10 based on the following criteria:
1) The summary should be concise, ideally within {ideal_length} sentences.
2) The summary should capture the main point of the article. Do not penalize if it doesn't include any additional concerns.
3) The summary should be an extractive summary, i.e., the sentences should be directly taken from the article without any modifications or removal of symbols.
4) The response should contain only the numerical rating.
5) Check your rating very carefully before responding.

### Response:
"""


LLAMA_SCORE_TEMPLATE = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Given the following document:
{document}

Provided Summary:
{summary}

Follow these instructions when writing your response:
* On a scale of 1-10, provide a numerical rating for the provided summary, with 10 denoting that the provided answer perfectly surmises the main topics of the document.
* Your response should contain only the numerical rating. DONOT include anything else like the provided answer, the ground truth answer, or an explanation of your rating scale in your response.
* Wrap your numerical rating inside <rating></rating> tags.
* Check very carefully before answering.
* Follow the output format as shown in the example below:
Example response:
<rating>7</rating>

### Response:
"""

LLAMA_GEN_TEMPLATE = """
### Instruction:
You are an expert data generator tasked with synthesizing new documents for a summarization task. {dataset_description}

Below are some example documents and their summaries from a group in the dataset (group 1):
{gp1_documents}

Below are some example documents and their summaries from another group in the dataset (group 2):
{gp2_documents}

Given the above documents, follow these instructions:
* synthesize a new document that follows a similar format to the examples provided.
* The document should contain {document_size}.
* The document should be coherent and relevant to the topic.
* The document should be original and not copied from the examples.
* Ensure that the document covers {alpha}% topics from group 1 and 1 - {alpha}% topics from group 2.
* Wrap your response in the <document></document> tags.
* Do NOT include anything else like the examples in your output.
"""