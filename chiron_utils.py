import spacy

nlp = spacy.load("en_core_web_lg") # can use md if you want, we use lg since we use it for masked character prediction as well

import re
from typing import List

regexspace = re.compile(r"\ +", re.IGNORECASE)
regexlines = re.compile(r"(\n(\ )?)+", re.IGNORECASE)


def get_sentences(text: str) -> List[str]:
    """split text into sentences with some minor formatting

    :param str text: input text
    :return List[str]: list of sub-sentences
    """
    # not great but fine system using spacy
    doc = nlp(text)
    sents = []
    for sent in doc.sents:
        # some models add "* " for bullets, just treat them as sentences - IGNORING NOW
        # sentence_text = sent.text.strip().replace("* ", "").replace("*", "")
        sentence_text = sent.text.strip()
        if len(sentence_text) == 0:
            continue
        for line in sentence_text.split("\n"):
            # sometimes llama has "Sure!"" sentences
            if line not in ["Sure!", "Sure, I'd be happy to help!"]:
                sents.append(line)

    return sents

def format_snippet_for_chiron_generation(snippet: str) -> str:
    """format the story section/snippet section of the CHIRON-generation prompt

    :param str snippet: the story snippet of interest
    :return str: section of the CHIRON Generation Module's prompt with the story snippet
    """

    template = """Story Section:
{snippet}"""

    nice_snippet = snippet.strip()

    nice_snippet = regexspace.sub(" ", nice_snippet)
    nice_snippet = regexlines.sub("\n", nice_snippet)

    snippet_text = template.format(snippet=nice_snippet)
    return snippet_text

def format_questions_and_answer_messages(qsandas: List) -> List:
    """takes a list of questions and answers, and puts them into messages for chat_template, prepending "Question" and "Answer" tags

    :param List qsandas: list of tuples of questions and answers
    :return List: list of formatted messages
    """
    messages = []
    for q, a in qsandas:
        q_message = {"role": "user", "content": "Question: " + q}
        a_message = {"role": "assistant", "content": "Answer: " + a}
        messages.append(q_message)
        messages.append(a_message)
    return messages