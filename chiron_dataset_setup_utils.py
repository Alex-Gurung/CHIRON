from chiron_utils import (
    format_questions_and_answer_messages,
    format_snippet_for_chiron_generation,
)


VERIFICATION_MODULE_ROLE_PROMPT = "You are a helpful and expert writing assistant. You will be given a section of a story or screenplay from the perspective of {character}. Please answer the following questions about the given statements and their relationship with the snippet provided."


def format_verification_prompt(
    CHECKPOINT_NAME,
    TOKENIZER,
    character,
    story_section,
    statement,
    cur_question,
    prior_questions_and_answers,
) -> str:
    """Formats a prompt for the verification module.

    :param str CHECKPOINT_NAME: The name of the language model checkpoint being used.
    :param tokenizer TOKENIZER: The tokenizer object for the language model.
    :param str character: The character's name or identifier.
    :param str story_section: The section of the story or screenplay to analyze.
    :param str statement: The statement to be verified against the story section.
    :param str cur_question: The current question to be answered.
    :param list prior_questions_and_answers: A list of tuples containing previous questions and their answers.
    :return str: A formatted prompt string for the verification module.
    """
    # llama has a separate section for roles
    using_system_role = "Llama" in CHECKPOINT_NAME

    # Note: prior_question_data should start with \n, everything else should be stripped
    role = VERIFICATION_MODULE_ROLE_PROMPT.format(character=character)
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

    new_messages = []
    for m in messages:
        new_m = m.copy()
        if m["role"] == ["user"]:
            new_m["content"] = new_m["content"] + " "
        new_messages.append(new_m)
    messages = new_messages

    tokenized_chat = TOKENIZER.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    # tokenized_chat += "Answer: "
    tokenized_chat += "Accuracy: "
    return tokenized_chat


def get_final_entailment_question():
    """Returns the final entailment question for the thoughtchain process.

    :return str: The final entailment question template string.
    """    
    # SHOULD CHECK TO MAKE SURE THAT THIS ACTUALLY IS THE RIGHT ONE - NOT SURE IF WE WERE
    # USING THE RIGHT QUESTION
    return """Rate the accuracy of provided statement about {character} on a scale of 1-5, where 1 is entirely inaccurate or unsupported and 5 is entirely accurate. If there are no claims made in the statement, mark the consistency of the statement as 1 as there is no evidence for the statement."""


def to_short_response(inf_or_amb_resp: str) -> str:
    """Converts a full inference or ambiguity response to a short answer format.

    :param str inf_or_amb_resp: The full response string from an inference or ambiguity question.
    :return str: The extracted short answer, typically a number or brief phrase.
    """
    return (
        inf_or_amb_resp.split("[/INST]")[-1]
        .split("Answer: ")[-1]
        .replace("</s>", "")
        .strip()
    )


def find_datapoint(
    dataset_mappings, snippet, statement, character, input_data, input_name
):
    """Searches for a specific datapoint in the given input dataset based on the provided snippet, statement, and character.

    :param dict dataset_mappings: A dictionary containing mappings of dataset entries
    :param str snippet: The snippet to search for
    :param str statement: The statement to search for
    :param str character: The character to search for
    :param list input_data: The input dataset to search through
    :param str input_name: The name of the input dataset
    :return dict: The matching datapoint if found
    :raises ZeroDivisionError: If the datapoint is not found in the dataset
    """
    scs = (snippet, character, statement)
    if scs in dataset_mappings[input_name]:
        return dataset_mappings[input_name][scs]
    for datapoint in input_data:
        dat_snippet, dat_sentence_of_interest = (
            datapoint["snippet"],
            datapoint["sentence_of_interest"],
        )
        dat_character = datapoint["character"]

        if (
            dat_snippet == snippet
            and dat_sentence_of_interest == statement
            and dat_character == character
        ):
            return datapoint
    # this should never happen, likely caused by creating datasets from different sources
    print(f"COULDNT FIND IN: {input_name}")
    print(snippet)
    print(character)
    print(statement)
    x = 1 / 0


def get_combined_data(
    CHECKPOINT_NAME,
    tokenizer,
    dataset_mappings,
    possible_snippet_character_statements,
    input_chain_data=None,
    input_inf_data=None,
    input_amb_data=None,
):
    """Creates a combined dataset by aligning different input datasets.

    This function iterates over possible snippet-character-statement combinations,
    finds relevant datapoints from each input dataset (chain, informative, and ambiguous),
    extracts required information, and combines them to create unified datapoints.

    :param str CHECKPOINT_NAME: Name of the checkpoint
    :param tokenizer: Tokenizer object
    :param dict dataset_mappings: Mappings of dataset entries
    :param list possible_snippet_character_statements: List of possible combinations
    :param list input_chain_data: Input chain dataset, defaults to None
    :param list input_inf_data: Input informative dataset, defaults to None
    :param list input_amb_data: Input ambiguous dataset, defaults to None
    :return list: Combined datapoints
    """
    # The function creates a combined dataset by aligning different input datasets
    # (chain, informative, and ambiguous). The resulting combined datapoints are
    # stored in a list and returned.
    temp_data = []
    # for datapoint in input_data:
    for (
        snippet,
        character,
        sentence_of_interest,
    ) in possible_snippet_character_statements:
        # should just be one question, need to add the other
        prior_questions_and_answers = []

        chain_datapoint = find_datapoint(
            dataset_mappings,
            snippet,
            sentence_of_interest,
            character,
            input_chain_data,
            "chain",
        )
        # chain information will be last but for convenience set it as the list
        prior_questions_and_answers = chain_datapoint["prior_entailment_qa"]

        prior_questions_and_answers.append(
            [
                chain_datapoint["entailment_question_1"],
                chain_datapoint["entailment_response_1"].replace("Answer: ", ""),
            ]
        )

        # both informative and ambiguous use ICL so the prior_qas are full of fake responses,
        # we just get the last one and put it at the beginning of the list

        inf_datapoint = find_datapoint(
            dataset_mappings,
            snippet,
            sentence_of_interest,
            character,
            input_inf_data,
            "inf",
        )
        inf_question = inf_datapoint["is_informative_question"].format(
            character=character
        )
        inf_response = to_short_response(inf_datapoint["is_informative_response"])
        inf_qa = [inf_question, inf_response]
        prior_questions_and_answers = [inf_qa, *prior_questions_and_answers]

        amb_datapoint = find_datapoint(
            dataset_mappings,
            snippet,
            sentence_of_interest,
            character,
            input_amb_data,
            "amb",
        )
        amb_question = amb_datapoint["is_ambiguous_question"].format(
            character=character
        )
        amb_response = to_short_response(amb_datapoint["is_ambiguous_response"])
        amb_qa = [amb_question, amb_response]
        prior_questions_and_answers = [amb_qa, *prior_questions_and_answers]

        question = get_final_entailment_question()

        cur_prompt = format_verification_prompt(
            CHECKPOINT_NAME,
            tokenizer,
            character,
            snippet,
            sentence_of_interest,
            question,
            prior_questions_and_answers,
        )

        # HACKY BUT TO MAKE SURE LAST SECTION SAYS SOMETHING DIFFERENT
        # we do this so we can train on completions, and only the final completions
        cur_prompt = cur_prompt.replace(
            f"{question[-5:]} [/INST]Answer: ", f"{question[-5:]} [/INST]Accuracy: "
        )

        temp_data.append([cur_prompt])

    return temp_data
