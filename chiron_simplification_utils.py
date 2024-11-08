from vllm import SamplingParams
from transformers import AutoTokenizer
from typing import List

from tqdm import tqdm

import re

from difflib import SequenceMatcher

from chiron_utils import get_sentences

regexspace = re.compile(r"\ +", re.IGNORECASE)
regexlines = re.compile(r"(\n(\ )?)+", re.IGNORECASE)

SIMPLIFICATION_ICL_BASE_QUERY = """Given the provided sentence, please split all independent clauses into independent sentences and resolve any issues with unclear pronouns or references. Only do this for compound sentences. Every new sentence should make sense on its own. Write them out in paragraph form, one sentence after another.

Sentence: She's curious about a closed door in Maxim's apartment and feels a strong urge to discover what's behind it.
Split Sentences: She's curious about a closed door in Maxim's apartment. She feels a strong urge to discover what's behind the closed door in Maxim's apartment.

Sentence: Kaluros is determined and focused during battles, using his magic and weapons effectively to defeat his enemies.
Split Sentences: Kaluros is determined and focused during battles, using his magic and weapons effectively to defeat his enemies.

Sentence: Hassan encountered a crab monster and engaged in a card battle to defeat it.
Split Sentences: Hassan encountered a crab monster. Hassan engaged in a card battle to defeat the crab monster.

Sentence: She uses imperatives to give orders and asks direct questions to gather information.
Split Sentences: She uses imperatives to give orders. She asks direct questions to gather information.

Sentence: Bob is easily distracted and forgets about the chase when he notices something outside.
Split Sentences: Bob is easily distracted. Bob forgets about the chase when he notices something outside.

Sentence: Rachel enters the warehouse to join the baby dragon, defying her initial skepticism.
Split Sentences: Rachel enters the warehouse to join the baby dragon, defying her initial skepticism.

Sentence: He gives commands to his companions and asks for their assistance.
Split Sentences: He gives commands to his companions. He asks for his companions' assistance.

Sentence: She explores the Zombear's massive body and climbs on it.
Split Sentences: She explores the Zombear's massive body. She climbs on the Zombear.

Sentence: Jordan opens the locker to find a locket, a newspaper, and a mysterious photograph.
Split Sentences: Jordan opens the locker to find a locket, a newspaper, and a mysterious photograph.

Sentence: He is quiet and tosses a gold idol between his hands while they wait for rescue.
Split Sentences: He is quiet. He tosses a gold idol between his hands while they wait for rescue.

Sentence: Timothy wants to jump up and down and sing a little song.
Split Sentences: """

def flatten_generation_outputs(GENERATION_OUTPUTS, CHOSEN_STORYID_CHARACTERS_SET=None):
    FLATTENED_GENERATION_OUTPUTS_PER_SENTENCE = []
    for output_data in GENERATION_OUTPUTS:
        if CHOSEN_STORYID_CHARACTERS_SET is not None:
            sid_char = (output_data["story_id"], output_data["character"])
            if sid_char not in CHOSEN_STORYID_CHARACTERS_SET:
                continue

        sentences_in_original_response = get_sentences(output_data["response"].strip())
        sentences_to_entail = sentences_in_original_response

        for sentence in sentences_to_entail:
            new_output_data_for_simplification = output_data.copy()
            # add a field for this specific sentence, this is the sentence we want to simplify
            new_output_data_for_simplification["sentence_of_interest"] = sentence

            FLATTENED_GENERATION_OUTPUTS_PER_SENTENCE.append(
                new_output_data_for_simplification
            )
    return FLATTENED_GENERATION_OUTPUTS_PER_SENTENCE

def run_simplification_task_over_model_with_custom_lengths(
    queries: List, sentences: List[str], model, NUM_BEAMS: int, tokenizer: AutoTokenizer
) -> List[str]:
    """Run the simplification task over the model. We need run each one through the model
    individually since we set the sampling parameters around the original length
    (to try not to change the sentences too much)

    :param List queries: the ICL prompts for simplification
    :param List[str] sentences: the original sentence (to get the tokenized sentence length)
    :param _type_ model: the LLM
    :param int NUM_BEAMS: sampling parameter
    :param AutoTokenizer tokenizer: corresponding tokenizer for sentence length
    :return List[str]: list of simplified outputs from the model
    """
    print(f"working on queries: {len(queries)}")
    text_per_prompt = []
    for index, inputs in tqdm(enumerate(queries), total=len(queries)):
        max_len = len(tokenizer(sentences[index])["input_ids"]) + 20
        min_token = max(0, max_len - 30)
        sampling_params = SamplingParams(
            min_tokens=min_token,
            max_tokens=max_len,
            best_of=NUM_BEAMS,
            stop=["\n"],
        )

        outputs = model.generate(
            sampling_params=sampling_params, prompt_token_ids=inputs.tolist()
        )

        generated_text = outputs[0].outputs[0].text
        # generated_text = outputs[0].sequences[0].text

        text_per_prompt.append(generated_text)

    return text_per_prompt


def get_chiron_simplification_format_prompt(
    tokenizer: AutoTokenizer,
    stats_and_simps: List,
    cur_statement: str,
    tokenize: bool = True,
):
    # Note: prior_question_data should start with \n, everything else should be stripped
    role = "You are an expert writing assistant helping an author split compound sentences. Please answer the following questions to the best of your ability."
    # question = "Given the following sentence, please split all clauses into their own independent sentences and resolve any issues with unclear pronouns or references. That is, every split sentence should staand completely on its own. Write them out in paragraph form, one sentence after another."
    question = "Given the provided sentence, please split all independent clauses into independent sentences and resolve any issues with unclear pronouns or references. Only do this for compound sentences. Every new sentence should make sense on its own. Write them out in paragraph form, one sentence after another."
    messages = []
    for stat, simps in stats_and_simps:
        # q_message = {"role": "user", "content": role + "\n" + stat}
        # q_message = {"role": "user", "content": question + "\n" + stat}
        q_message = {"role": "user", "content": stat}
        a_message = {"role": "assistant", "content": simps}
        # q_message = {"role": "user", "content": "Sentence: " + stat}
        # a_message = {"role": "assistant", "content": "Split Sentences: " + simps}
        messages.append(q_message)
        messages.append(a_message)
    messages[0]["content"] = role + "\n" + question + "\n" + messages[0]["content"]

    messages.append(
        # {"role": "user", "content": question + "\n" + "Sentence: " + cur_statement},
        {"role": "user", "content": "Sentence: " + cur_statement},
    )

    if tokenize:
        tokenized_chat = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
    else:
        tokenized_chat = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    return tokenized_chat


def reformat_sentences_from_simplification(
    all_data_to_save: List, output_texts: List[str]
):
    """Update all_data_to_save with the simplified sentences, replacing the 'response' field
    with the generated simplified sentences (filtered for overlap with the original input)

    :param List all_data_to_save: _description_
    :param List[str] output_texts: _description_
    """
    for sentence_index, gentext in enumerate(output_texts):

        all_data_to_save[sentence_index]["old_response"] = all_data_to_save[
            sentence_index
        ]["response"]
        all_data_to_save[sentence_index]["split_sentences_raw"] = gentext.strip()
        # Note: the vllm output is just the new text so we don't have to do any splitting
        # We assume all responses start with Split Sentences: If the response doesn't it would cause problems
        # but every generation so far has been fine
        # has_split = "Split Sentences:" in gentext
        # print(f"ORIG RESPONSE HAS SPLIT: {has_split}")
        nice_gentext = gentext.replace("Split Sentences:", "").replace("</s>", "")
        nice_gentext = nice_gentext.replace("Split Sentence:", "")
        nice_gentext = nice_gentext.strip()

        all_data_to_save[sentence_index]["split_sentences_list"] = get_sentences(
            nice_gentext
        )

        # check to make sure the new sentences overlap with the original
        # we want a high percentage of the new simplified sentence to already exist
        # in the original (so it's not an invented or unrelated sentence)

        soi = all_data_to_save[sentence_index]["sentence_of_interest"]
        split = all_data_to_save[sentence_index]["split_sentences_list"]
        new_response_sentences = []
        found_simplified = False
        for sen in split:
            if len(sen) == 0:
                continue
            # match = SequenceMatcher(None, soi, sen).find_longest_match()
            match = SequenceMatcher(None, soi, sen).find_longest_match(0, len(soi), 0, len(sen))
            percent = match.size / len(sen) * 100
            if percent > 80:
                found_simplified = True
                new_response_sentences.append(sen)
        if not found_simplified:
            new_response_sentences.append(soi)

        new_response = " ".join(new_response_sentences)
        all_data_to_save[sentence_index]["response"] = new_response
