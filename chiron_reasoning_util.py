from typing import List
from transformers import AutoTokenizer
from tqdm import tqdm
from chiron_utils import (
    format_questions_and_answer_messages,
    format_snippet_for_chiron_generation,
    get_sentences,
)


def get_icl_role_prompt() -> str:
    """Returns the role prompt for in-context learning (ICL) tasks.

    :return str: The role prompt for ICL tasks.
    """
    return "You are a helpful and expert writing assistant. Please answer the following questions to the best of your ability."


def format_icl_prompt(
    TOKENIZER: AutoTokenizer,
    character: str,
    statement: str,
    cur_question: str,
    prior_questions_and_answers: List,
    tokenize: bool = True,
    using_system_role: bool = False,
    return_messages: bool = False,
) -> List[int]:
    """Formats a prompt for in-context learning (ICL) tasks.

    This function prepares a prompt for ICL tasks by combining character information,
    a statement, the current question, and any prior questions and answers. It handles
    different formatting requirements based on whether a system role is used or not.

    :param AutoTokenizer TOKENIZER: _description_
    :param str character: _description_
    :param str statement: _description_
    :param str cur_question: _description_
    :param List prior_questions_and_answers: _description_
    :param bool tokenize: _description_, defaults to True
    :param bool using_system_role: some models (e.g. LLama) support a system role
    message type, defaults to False
    :param bool return_messages: whether to return the messages or the tokenized prompt, defaults to False. Overrides tokenize.
    :return List[int]: tokenized prompt
    """

    # Note: prior_question_data should start with \n, everything else should be stripped
    role = get_icl_role_prompt().format(character=character)
    statement_text = statement.strip()
    # prior_question_text = get_prior_question_text(prior_questions_and_answers)
    prior_question_messages = format_questions_and_answer_messages(
        prior_questions_and_answers
    )
    cur_question = cur_question.strip()
    user_message = None

    # add question message separately after question messages
    question_message = f"Question: {cur_question}\nStatement: {statement_text}".replace(
        "{character}", character
    )

    user_message = ""
    if not using_system_role:
        user_message = role

    # some of the text might have the {character} formatting string that still needs to be infilled
    user_message.format(character=character)

    if len(prior_question_messages) == 0:
        user_message += "\n\n" + question_message

    # format for the chat_template
    messages = []
    if using_system_role:
        messages = [
            {"role": "system", "content": role},
        ]
    messages.append(
        {"role": "user", "content": user_message},
    )

    # this and the if statement above are mutually exclusive
    # if there are prior messages then we add the question message separately
    # if there are not prior messages, then it's already in user_message
    # unfortunately we have to retcon the first question into the chat template
    if len(prior_question_messages) > 0:
        messages[-1]["content"] += "\n\n" + prior_question_messages[0]["content"]
        messages.extend(prior_question_messages[1:])
        messages.append({"role": "user", "content": question_message})

    if return_messages:
        return messages

    if tokenize:
        tokenized_chat = TOKENIZER.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
    else:
        tokenized_chat = TOKENIZER.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    return tokenized_chat


def reformat_icl_questions_and_answers_into_question_answer_pairs(
    icl_questions_and_answers: List[str],
):
    """Reformats in-context learning questions and answers into question-answer pairs.

    This function takes a string containing multiple ICL questions and answers,
    and reformats them into a list of tuples, where each tuple contains a question
    and its corresponding answer.

    :param List[str] icl_questions_and_answers: A list containing a single string
        with multiple ICL questions and answers in a specific format.
    :return List[Tuple[str, str]]: A list of tuples, where each tuple contains
        a question and its corresponding answer.
    """

    # icl_questions_and_answers should have format:
    #
    # Question: {question}
    # Statement: {statement}
    # Answer: {answer}
    # repeated with \n\n in between each example

    prior_questions_and_answers = []
    # convert ICL prior questions and answers into messages
    questions_and_answers = icl_questions_and_answers.split("\n\n")
    for qa in questions_and_answers:
        question_section, answer_section = qa.split("\nAnswer: ")
        question_section = question_section.split("Question: ")[1].strip()
        answer_section = answer_section.strip()
        prior_questions_and_answers.append((question_section, answer_section))
    return prior_questions_and_answers


def prepare_simplified_data_for_reasoning(SIMPLIFICATION_OUTPUTS: List):
    """Prepares simplified data for reasoning by flattening the simplification outputs.

    This function takes the output from a simplification process and flattens it into a list
    where each element represents a single sentence from the simplified responses.

    :param List SIMPLIFICATION_OUTPUTS: A list of dictionaries, each containing the results of a simplification process.
                                        Each dictionary is expected to have keys like 'response', 'sentence_of_interest', etc.
    :return List: A flattened list of dictionaries, where each dictionary represents a single sentence
                  from the simplified responses, along with its associated metadata.
    """
    # flatten the data from simplification
    FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE = []

    for query_data in tqdm(SIMPLIFICATION_OUTPUTS):
        # sentences to run entailment on
        sentences_in_simplified_response = get_sentences(query_data["response"].strip())

        for sentence in sentences_in_simplified_response:
            new_query_data_for_entailment = query_data.copy()
            new_query_data_for_entailment["simplified_sentence_of_interest"] = (
                new_query_data_for_entailment["sentence_of_interest"]
            )
            new_query_data_for_entailment["sentence_of_interest"] = sentence
            FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE.append(
                new_query_data_for_entailment
            )
    return FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE


def get_icl_based_prompts(
    tokenizer: AutoTokenizer,
    FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE: List,
    question: str,
    icl_question_answers: List[str],
    using_system_role: bool = False,
    return_messages: bool = False,
):
    """Generates prompts for in-context learning (ICL) based on simplified data.

    This function prepares prompts for each sentence in the flattened simplification outputs,
    incorporating given in-context learning examples and the given question.

    :param AutoTokenizer tokenizer: The tokenizer to be used for processing the text.
    :param List FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE: A list of dictionaries containing simplified sentence data.
    :param str question: The main question to be answered for each sentence.
    :param List[str] icl_question_answers: A list of strings containing in-context learning examples (questions and answers).
    :param bool using_system_role: Flag to indicate if a system role should be used in prompt formatting, defaults to False.
    :return List[List[int]]: A list of tokenized prompts, one for each sentence in the input data.
    """

    # get prompts for each sentence and this question
    all_prompts_for_this_question = []
    # iterate over per-sentence data
    for sentence_index, query_data in tqdm(
        enumerate(FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE),
        total=len(FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE),
    ):
        # get the snippet, sentence of interest, and character
        snippet, sentence_of_interest, character = (
            query_data["snippet"],
            query_data["sentence_of_interest"],
            query_data["character"],
        )

        prior_questions_and_answers = (
            reformat_icl_questions_and_answers_into_question_answer_pairs(
                icl_question_answers
            )
        )

        # get prompt - note this is now messages-based but needs to be a list for vllm
        cur_prompt = format_icl_prompt(
            tokenizer,
            character,
            sentence_of_interest,
            question,
            prior_questions_and_answers,
            tokenize=True,
            return_messages=return_messages,
            using_system_role=using_system_role,
        )
        if not return_messages:
            cur_prompt = cur_prompt.tolist()
            all_prompts_for_this_question.append(cur_prompt[0])
        else:
            all_prompts_for_this_question.append(cur_prompt)

    return all_prompts_for_this_question


def get_ambiguous_question() -> str:
    """Returns a question template for determining statement ambiguity.

    :return str: A question template for determining statement ambiguity.
    """
    return """Is the given statement about {character} ambiguous in a way that makes the meaning unclear? Ambiguities may include, but are not limited to, references to unspecified characters, objects, and actions. If the statement begins with a personal pronoun (e.g. \"He\" or \"She\"), assume it refers to {character} and don't count the pronoun towards the ambiguity. Begin your 1-2 sentence response with "Yes" if the statement is too ambiguous to be understood on its own and "No" if the statement makes is unambiguous in its meaning."""


def get_ambiguous_in_context_questions() -> str:
    """Returns a list of in-context questions and answers for determining if a statement is overly ambiguous.

    :return str: A list of (character names, statements, and corresponding ambiguity answers).
    """
    chars_statements_and_answers = [
        [
            "Collins",
            "No other physical descriptions of Collins are provided in this snippet.",
            "No, the statement is unambiguous in its meaning as there are no claims made.",
        ],
        [
            "Mustafa",
            "He has a strange glowing key that he uses to open the door to his home.",
            "No, the statement is unambiguous in its description of Mustafa's ownership of key and his opening of a door. The pronouns 'he' and 'his' unambiguously refer to Mustafa.",
        ],
        [
            "Kelly",
            "These men run away at first sight.",
            'Yes, the statement is ambiguous because it doesn\'t specify who "These men" are, and it is also unclear what "first sight" refers to.',
        ],
        [
            "Dr. Alex",
            "They are skittish and afraid of the darkness around their camp.",
            "No, the statement is unambiguous about their (Alex's) fear of the darkness.",
        ],
        [
            "Luis",
            "They struggle to stand up.",
            "No, the statement is unambiguous about Luis's difficulty standing up.",
        ],
        [
            "Arjun",
            "Arjun's primary goal is to resolve this situation.",
            'Yes, the statement is ambiguous because we cannot understand Arjun\'s goal without knowing "this situation".',
        ],
        [
            "Wang Fang",
            "She speaks softly to them, hoping to calm them down.",
            'Yes, the statement is ambiguous because it is unclear who "them" refers to, which is necessary to understand the statement.',
        ],
        [
            "Ping",
            'She speaks using colloquial expressions like "C\'est la vie" and "TGIF, am I right?".',
            "No, the statement is unambiguous in its description of Ping's speaking habits.",
        ],
        [
            "Santiago",
            "He is skinny with long legs.",
            "No, the statement is unambiguous in its description of Santiago.",
        ],
        [
            "Jarvis the Robot",
            "He opens a brown bottle and drinks the murky liquid inside.",
            "No the statement is unambiguous in its description of Jarvis drinking from the brown bottle.",
        ],
        [
            "The Bartender",
            "Her keen eyes spot Arthur sneaking the coin out from underneath the cup, showing her skills in observation.",
            "No the statement is unambiguous in its description of the Bartender's observational skills.",
        ],
        [
            "Mohammed",
            "He aims to accomplish this by using fire as a means to drive it away, based on his knowledge of myths and lore from various cultures.",
            "Yes, the statement is ambiguous as it is unclear what he is trying to accomplish or who/what he is driving aways.",
        ],
        [
            "Jake",
            "Overall, Jake seems to be focused on his social life and maintaining his reputation.",
            "No, the statement is unambiguous in its description of Jake's focus.",
        ],
        [
            "Pedro",
            "His accomplishes this goal quickly.",
            'Yes, the statement is ambiguous as it is unclear what "this goal" refers to.',
        ],
        [
            "Fadekemi",
            "Based on this story snippet, they have a mysterious notebook.",
            "No, the statement is unambiguous in describing what Fadekemi has.",
        ],
        [
            "Daniel the Destroyer",
            "Daniel seems very familiar with the inner workings of the ship.",
            "No, the statement is unambiguous in its description of Daniel's knowledge of the ship.",
        ],
        [
            "Merlin the Magician",
            "Based on this story snippet, he greatly values the guidance of others, as evidenced by him asking for Sunita's advice.",
            "No, the statement is unambiguous and describes Merlin's appreciation of other people's advice.",
        ],
        [
            "Abdul, Healer of the Ages",
            "They try very hard to speak with the accent of the High Aristocracy, but sometimes slip into less-idolized accent of his hometown.",
            "No, the statement is unambiguous and describes Abdul's accent.",
        ],
        [
            "Dmitry",
            "Based on the story section provided, we can infer that he speaks with confidence and assertiveness.",
            "No, the statement is unambiguous in its description of Dmitry's speaking pattern.",
        ],
        [
            "Hassan of Atlantis",
            "He is in love with Cassian.",
            "No, the statement is unambiguous in who Hassan loves.",
        ],
        [
            "Rockefeller Moneybags",
            "Furthermore, they are easily scared by the screams coming from the movie theatre.",
            "No, the statement is unambiguous in describing Rockefeller being scared by the screams.",
        ],
    ]

    full_text = []
    for c, s, a in chars_statements_and_answers:
        question = get_ambiguous_question()
        addition = f"""Question: {question}
Statement: {s}
Answer: {a}""".format(
            character=c
        )
        full_text.append(addition)
    return "\n\n".join(full_text)


def get_informative_question() -> str:
    """Returns a question template for determining if a statement provides any novel information about a character.

    :return str: A question template for determining if a statement provides any novel information about a character.
    """
    return """Does this statement give you any novel information concerning {character} or what {character} knows? Novel information may include, but is not limited to, physical descriptions, new information they may have learned, goals they have or actions they have just completed, and descriptions of their speech. Begin your response with "Yes" if the statement gives us any new information and "No" if the statement doesn't add to our knowledge/understanding of the character in any way."""


def get_informative_in_context_questions() -> str:
    """Returns a list of in-context questions and answers for determining if a statement provides any novel information about a character. (i.e. is the statement informative?)

    :return str: A list of (character names, statements, and corresponding answers) about the informative-ness of those statements.
    """
    chars_statements_and_answers = [
        [
            "Collins",
            "Collins learned that his father died after the factory accident in 1973.",
            "Yes, the statement provides information about what Collins has learned about his father's death.",
        ],
        [
            "Kelly",
            "There were no descriptions of Kelly.",
            "No, the statement does not give you any information about Kelly.",
        ],
        [
            "Ani Martirosyan",
            "She gestures wildly, and speaks with flowerly descriptions.",
            "Yes, the statement provides information on how Ani speaks.",
        ],
        [
            "Dr. Alex",
            "They have a new goal of finding a place to eat.",
            "Yes, the statement is gives us information about Dr. Alex's new goal.",
        ],
        [
            "Rockefeller Moneybags",
            "This snippet provides many physical descriptions of Rockefeller.",
            "No, the statement does not provide any information about Rockefeller.",
        ],
        [
            "The Bartender",
            "She is an extremely focused and driven individual.",
            "Yes, the statement is describes the Bartender's personality.",
        ],
        [
            "Arjun",
            "Jennifer aims to accomplish this by using fire as a means to drive it away, based on her knowledge of myths and lore from various cultures.",
            "No, the statement does not make any claims about Arjun as it only makes claims about Jennifer.",
        ],
        [
            "Wang Fang",
            "She expresses a desire to look into the matter further and gather more information, indicating that she is motivated to uncover the truth about the space/time anomaly and its potential impact on the city's population.",
            "Yes, the statement describes her desire to investigate.",
        ],
        [
            "Santiago",
            "He is skinny with long legs.",
            "Yes, the statement describes his physical appearance.",
        ],
        [
            "Merlin the Magician",
            "According to this section of the story, Merlin's primary goal is to live to see the night.",
            "Yes, the statement describes describes Merlin's goal to survive.",
        ],
        [
            "Mohammed",
            "His internal motivations do not significantly change in this snippet.",
            "No, the statement does not provide any new information about Mohammed.",
        ],
        [
            "Isabella",
            "She recently acquired a gun while searching the trunks abandoned in the cave.",
            "Yes, the statement tells the reader a fact about her recent activity and a new item she possesses.",
        ],
        [
            "Pedro",
            "Here are some physical descriptions of Pedro based on the given story section:",
            "No, the statement does not provide any information about Pedro.",
        ],
        [
            "Fadekemi",
            "They are a tall warrior.",
            "Yes, the statement describes their height and profession.",
        ],
        [
            "Ava the Antagonist",
            "Here are some descriptions of how Ava speaks:",
            "No, the statement does not provide any information about Ava, although it implies that following statements will contain descriptions of how they speak.",
        ],
        [
            "Ying Li",
            "However, we learned that they speak brashly and with profanity, with little regard for others.",
            "Yes, the statement describes their speaking style.",
        ],
        [
            "Emmanuel the Destroyer",
            "He carries a med-kit and a set of knives in his backpack, just in case the werewolves show up.",
            "Yes, the statement gives us new information about what Emmanuel the Destroyer has in his bag.",
        ],
        [
            "Abdul, Healer of the Ages",
            "A wide grin slowly grows across their face.",
            "Yes, the statement describes Abul grinning.",
        ],
        [
            "Dmitry",
            "Based on the story section provided, we learn that he has a mysterious book of magic.",
            "Yes, the statement gives us the information that Dmitry has a book of magic.",
        ],
        [
            "Hassan of Atlantis",
            "He does not handle stress very well.",
            "Yes, the statement describes his inability to handle stress.",
        ],
    ]
    full_text = []
    for c, s, a in chars_statements_and_answers:
        question = get_informative_question()
        addition = f"""Question: {question}
Statement: {s}
Answer: {a}""".format(
            character=c
        )
        full_text.append(addition)
    return "\n\n".join(full_text)


def get_thoughtchain_role_prompt() -> str:
    """Returns the role prompt for the thoughtchain reasoning steps.

    :return str: A string containing the role prompt for the thoughtchain reasoning steps.
    """    
    return "You are a helpful and expert writing assistant. You will be given a section of a story or screenplay from the perspective of {character}. Please answer the following questions about the given statements and their relationship with the snippet provided."


def get_entailment_questions() -> List[str]:
    """Returns a list of entailment questions for the thoughtchain reasoning process.

    :return List[str]: A list containing two entailment questions as strings.
    """    
    q1 = """What, if any, section of the story snippet is most relevant to the given statement? Provide a brief 1-2 sentence description of this section or "N/A" if there is no relevant section."""
    q2 = """In 1-2 sentences, compare the claim the statement makes and the section of story you highlighted in your previous answer. Are there any notable differences? Are all claims made by the statement explicitly supported? If there are no claims, write "N/A"."""
    return [q1, q2]


def format_thoughtchain_prompt(
    tokenizer: AutoTokenizer,
    character: str,
    story_section: str,
    statement: str,
    cur_question: str,
    prior_questions_and_answers: List,
    using_system_role: bool = False,
    return_messages: bool = False,
) -> str:
    """Formats a prompt for the thought-chain reasoning process.

    This function creates a formatted prompt for the thought-chain reasoning process,
    incorporating the given story section, character, statement, and questions.

    :param AutoTokenizer tokenizer: The models' tokenizer.
    :param str character: The character's name.
    :param str story_section: The relevant section of the story.
    :param str statement: The statement about the character (based on the story section).
    :param str cur_question: The current question to be answered in the reasoning process.
    :param List prior_questions_and_answers: A list of tuples containing previous reasoning steps (e.g. informative and ambiguous questions).
    :param bool using_system_role: Flag to determine if a system role should be used in the prompt, defaults to False.
    :param bool return_messages: Flag to determine if the function should return the messages or the tokenized prompt, defaults to False.
    :return str: A formatted prompt for the thought-chain reasoning process.
    """    
    # Note: prior_question_data should start with \n, everything else should be stripped
    role = get_thoughtchain_role_prompt().format(character=character)
    statement_text = statement.strip()
    story_section_text = format_snippet_for_chiron_generation(story_section)
    # prior_question_text = get_prior_question_text(prior_questions_and_answers)
    prior_question_messages = format_questions_and_answer_messages(
        prior_questions_and_answers
    )
    cur_question = cur_question.strip()
    user_message = None

    # add question message separately after question messages
    question_message = f"Question: {cur_question}".format(character=character)

    if using_system_role:
        prior_to_questions_template = """{story_section_text}

Please answer the following questions about {character} by comparing the provided statement with the story section above:

Statement: {statement_text}"""
        user_message = prior_to_questions_template.format(
            story_section_text=story_section_text,
            statement_text=statement_text,
            character=character,
        )
    else:
        prior_to_questions_template = """{role}

{story_section_text}

Please answer the following questions about {character} by comparing the provided statement with the story section above:

Statement: {statement_text}"""
        user_message = prior_to_questions_template.format(
            role=role,
            story_section_text=story_section_text,
            statement_text=statement_text,
            character=character,
        )

    try:
        # some of the text might have the {character} formatting string that still needs to be infilled
        user_message = user_message.format(character=character)
    except:
        # this error arises when user_message contains some brackets from the story
        pass

    if len(prior_question_messages) == 0:
        user_message += "\n\n" + question_message

    # format for the chat_template
    messages = []
    if using_system_role:
        messages = [
            {"role": "system", "content": role},
        ]
    messages.append(
        {"role": "user", "content": user_message},
    )

    # this and the if statement above are mutually exclusive
    # if there are prior messages then we add the question message separately
    # if there are not prior messages, then it's already in user_message
    # unfortunately we have to retcon the first question into the chat template
    if len(prior_question_messages) > 0:
        messages[-1]["content"] += "\n\n" + prior_question_messages[0]["content"]
        messages.extend(prior_question_messages[1:])
        messages.append({"role": "user", "content": question_message})

    if return_messages:
        # note: this doesn't add the generation prompt or "Answer:" prefix
        return messages

    tokenized_chat = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    tokenized_chat += "Answer: "
    return tokenized_chat
