import re
from transformers import AutoTokenizer
from typing import List, Tuple
from chiron_utils import format_snippet_for_chiron_generation

regexspace = re.compile(r"\ +", re.IGNORECASE)
regexlines = re.compile(r"(\n(\ )?)+", re.IGNORECASE)

CHIRON_GENERATION_ROLE_PROMPT = "You are a helpful and expert writing assistant. You will be given a section of a story or screenplay. Please answer the following questions about the character learned in this story section, and respond in short paragraph form."

def get_chiron_generation_question_text(question: str, character: str) -> str:
    if question == "" or question is None:
        return ""
    question = question.strip().replace("\n", " ")
    template = """Please answer the following questions about {character} with short, succinct sentences based on the given story section.

Question: {question} Respond in paragraph form with short, simple sentences with no dependent clauses or transition words.
Answer:"""
    question_text = template.format(question=question, character=character)
    return "\n\n" + question_text


def format_chiron_generation_prompt(
    tokenizer: AutoTokenizer,
    story_snippet: str,
    character: str,
    question: str,
    using_system_role: bool = False,
):
    """returns the prompt for CHIRON's generation module, given a story snippet, character, and question

    :param AutoTokenizer tokenizer: the tokenizer to use for the prompt, should correspond to the model
    and the using_system_role parameter
    :param str story_snippet: the snippet of story to base CHIRON on
    :param str character: the character of interest
    :param str question: the CHIRON question used to elicit character information
    :param bool using_system_role: some models (e.g. LLama) support a system role
    message type, defaults to False
    """

    role = CHIRON_GENERATION_ROLE_PROMPT

    story_snippet_text = format_snippet_for_chiron_generation(story_snippet)
    question_text = get_chiron_generation_question_text(question, character)

    user_message = None
    if using_system_role:
        template = """{story_snippet_text}{question_text}"""
        user_message = template.format(
            story_snippet_text=story_snippet_text,
            character=character,
            question_text=question_text,
        )
    else:
        template = """{role}
{story_snippet_text}{question_text}"""
        user_message = template.format(
            role=role,
            story_snippet_text=story_snippet_text,
            character=character,
            question_text=question_text,
        )

    messages = []
    if using_system_role:
        messages = [
            {"role": "system", "content": role},
        ]
    messages.append(
        {"role": "user", "content": user_message},
    )
    tokenized_chat = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    return tokenized_chat


def get_dialogue_qs() -> Tuple[str, List[str]]:
    """returns the title and questions for the dialogue section of CHIRON

    :return Tuple[str, List[str]]: [title, [question_1, question_2, etc.]]
    """

    q1 = "What, if anything, have we learned about how this character speaks from this snippet?"
    qs = [q1]
    return "## Character Dialogue", qs


def get_physical_qs() -> Tuple[str, List[str]]:
    """returns the title and questions for the physical/personality section of CHIRON

    :return Tuple[str, List[str]]: [title, [question_1, question_2, etc.]]
    """

    q1 = "What, if any, physical descriptions of this character are in this snippet?"
    q2 = "What, if any, descriptions of this character's personality are in this snippet?"
    qs = [q1, q2]
    return "## Personality/Physical Attributes", qs


def get_knowledge_qs() -> Tuple[str, List[str]]:
    """returns the title and questions for the knowledge section of CHIRON

    :return Tuple[str, List[str]]: [title, [question_1, question_2, etc.]]
    """
    q1 = "What, if any, factual information is given about this character in this snippet?"
    q2 = "What, if any, information has this character learned in this snippet?"
    qs = [q1, q2]
    return "## Knowledge", qs


def get_plot_qs() -> Tuple[str, List[str]]:
    """returns the title and questions for the plot section of CHIRON

    :return Tuple[str, List[str]]: [title, [question_1, question_2, etc.]]
    """
    q1 = "What, if any, goals does this character gain in this snippet that they wish to accomplish in the future?"
    q2 = "What, if any, goals does this character complete in this snippet?"
    q3 = "How, if at all, does this character's internal motivations change in this snippet?"

    qs = [q1, q2, q3]
    return "## Plot and Motivation", qs


CHIRON_TITLE_QUESTION_SETS = [
    get_physical_qs(),
    get_dialogue_qs(),
    get_knowledge_qs(),
    get_plot_qs(),
]


def get_prompt_data_for_chiron_generation(
    tokenizer: AutoTokenizer,
    snippet: str,
    character: str,
    snippet_role: str = None,
    using_system_role: bool = False,
    story_id: str = "storyid",
):
    """returns all the prompts necessary for the Generation Module in CHIRON, packaged
    in a dictionary with other information we might want to save

    :param AutoTokenizer tokenizer: the tokenizer to use for the prompt, should correspond to the model
    and the using_system_role parameter
    :param str snippet: the story snippet of interest
    :param str character: the character of interest
    :param str snippet_role: some datasets (e.g. storium) have roles per snippet
    that are useful to keep track of, defaults to None
    :param bool using_system_role: some models (e.g. LLama) support a system role
    message type, defaults to False
    """

    all_prompt_data_for_snip_char = []
    # iterate over each of the question sets
    for qsid, title_question_set in enumerate(CHIRON_TITLE_QUESTION_SETS):
        title, question_set = title_question_set
        # iterate over each question
        for qnum, question in enumerate(question_set):
            cur_prompt = format_chiron_generation_prompt(
                tokenizer,
                snippet,
                character,
                question,
                using_system_role=using_system_role,
            )
            all_prompt_data_for_snip_char.append(
                {
                    "snippet_role": snippet_role,
                    "prompt": cur_prompt,
                    "character": character,
                    "snippet": snippet,
                    "question": question,
                    "question_set": title,
                    "story_id": story_id,
                    
                }
            )
    return all_prompt_data_for_snip_char