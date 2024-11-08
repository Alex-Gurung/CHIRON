############################################################################################
# GENERAL STRUCTURE OF CHIRON GENERATION
#
# 1) Generate basic structure
# 2) Simplify (technically optional, but recommended to separate sentences into atomic claims)
# 3) Generate ambiguous/informative/chain of thought
# 4) Feed those into verification classifier
# 5) Compile character sheet, filtering using verification classifier's predictions
############################################################################################

import argparse
import json
import os

import jsonlines
from vllm import LLM, SamplingParams

from chiron_dataset_setup_utils import get_combined_data
from chiron_filter_utils import get_model_predictions_across_dataset
from chiron_generation_module_utils import get_prompt_data_for_chiron_generation

from transformers import AutoTokenizer, AutoModelForCausalLM
from chiron_simplification_utils import (
    SIMPLIFICATION_ICL_BASE_QUERY,
    flatten_generation_outputs,
    get_chiron_simplification_format_prompt,
    reformat_sentences_from_simplification,
    run_simplification_task_over_model_with_custom_lengths,
)
from chiron_reasoning_util import (
    format_thoughtchain_prompt,
    get_ambiguous_in_context_questions,
    get_ambiguous_question,
    get_entailment_questions,
    get_icl_based_prompts,
    get_informative_in_context_questions,
    get_informative_question,
    prepare_simplified_data_for_reasoning,
)

from chiron_compile_character_sheet_utils import (
    fast_new_output_predictions_from_simpified_and_original_outputs,
    get_character_sheet_for_one_snippet_data,
)
import torch

import pandas as pd
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Chiron generation pipeline")

    # Model parameters
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HuggingFace model checkpoint (default: mistralai/Mistral-7B-Instruct-v0.2). Paper uses Mistral-7B-Instruct-v0.2 for downstream tasks, but other models are supported (tested with llama 3.1).",
    )

    # Input parameters
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="JSON file containing snippet, character, and snippet_role. If not provided, example values are used. `snippet_role` is the name of the character whose perspective is being described in the snippet. This comes from STORIUM and is almost always the same as character, but can be set as a different value if the chapter is from a different character's perspective.",
    )

    parser.add_argument(
        "--tempfile-save-dir",
        type=str,
        default="tmp/",
        help="Directory to save temporary files to (default: tmp/). Will create if it does not exist.",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file name. If not provided, character sheet is not saved (just printed to console).",
    )

    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Maximum tokens for generation (default: 150). Also used for simplification.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Number of beams for generation (default: 4). Also used for simplification.",
    )

    # Reasoning parameters (used for ambiguous/informative/thought chain)
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        default=75,
        help="Maximum tokens for reasoning tasks (default: 75)",
    )
    parser.add_argument(
        "--reasoning-num-beams",
        type=int,
        default=4,
        help="Number of beams for reasoning tasks (default: 4)",
    )

    # Verification model
    parser.add_argument(
        "--verification-model",
        type=str,
        default="agurung/character_sheet_entailment_model___mistral7bv2",
        help="Verification model checkpoint (default: agurung/character_sheet_entailment_model___mistral7bv2). Using a different model may result in worse performance (especially with regard to precision)",
    )

    return parser.parse_args()


def get_input_data(args):
    # Default values
    default_data = {
        "snippet": """Nadia watched both of the men silently, pulling her tattered, too-big sweatshirt tighter around herself. She hadn't spoken since the plane had crashed, and she didn't want to, but it looked like the men were about to devolve into an argument about whether or not to go into the cave, and she actually had input.  She cleared her throat and, when she had the attention of both Jacob and Jayson, she pointed to the sky. Their heads all turned.  They could barely see them through the trees, but the grey storm clouds were unmistakable all the same.  Nadia tried to speak but had to clear her throat once more and swallow before she was able to do so, not having used her voice for so long. "I don't know about you, but I'd really rather not be out in the forest during a storm. I say we check out the cave. If we're careful about it, we won't run into any trouble.\"""",
        "character": "Nadia",
        "snippet_role": "Nadia",
    }

    if args.input_file and os.path.exists(args.input_file):
        try:
            with open(args.input_file, "r") as f:
                data = json.load(f)
                return (
                    data.get("snippet", default_data["snippet"]),
                    data.get("character", default_data["character"]),
                    data.get("snippet_role", default_data["snippet_role"]),
                )
        except json.JSONDecodeError:
            print(f"Error reading {args.input_file}, using default values")
            return (
                default_data["snippet"],
                default_data["character"],
                default_data["snippet_role"],
            )

    return (
        default_data["snippet"],
        default_data["character"],
        default_data["snippet_role"],
    )


regexspace = re.compile(r"\ +", re.IGNORECASE)
regexlines = re.compile(r"(\n(\ )?)+", re.IGNORECASE)


def main():

    args = parse_arguments()
    
    # create the tempfile save dir if it does not exist
    if not os.path.exists(args.tempfile_save_dir):
        os.makedirs(args.tempfile_save_dir)

    snippet, character, snippet_role = get_input_data(args)

    ##########################################################################
    # Generation Module
    ##########################################################################

    # Update model parameters based on args
    GENERATION_MODULE_CHECKPOINT_NAME = args.checkpoint
    # Create a shorter model name from the checkpoint (everything after the last slash)
    GENERATION_MODULE_MODEL_NAME = GENERATION_MODULE_CHECKPOINT_NAME.split("/")[
        -1
    ].lower()

    GENERATION_MODULE_MAX_TOKENS = args.max_tokens
    GENERATION_MODULE_NUM_BEAMS = args.num_beams

    # We support using the system role message type, but as we primarily test with mistral and llama models, we just check for that
    USING_SYSTEM_ROLE = "llama" in GENERATION_MODULE_CHECKPOINT_NAME.lower()

    print(
        f"Loading generation model and tokenizer: {GENERATION_MODULE_CHECKPOINT_NAME}"
    )
    model = LLM(model=GENERATION_MODULE_CHECKPOINT_NAME)
    print(f"Model loaded!")
    TOKENIZER = AutoTokenizer.from_pretrained(GENERATION_MODULE_CHECKPOINT_NAME)
    TOKENIZER.padding_side = "right"
    TOKENIZER.add_special_tokens({"pad_token": "[PAD]"})
    print(f"Tokenizer loaded!")

    GENERATION_MODULE_SAMPLING_PARAMS = SamplingParams(
        max_tokens=GENERATION_MODULE_MAX_TOKENS,
        best_of=GENERATION_MODULE_NUM_BEAMS,
        stop=["\n"],
    )

    all_prompt_data = get_prompt_data_for_chiron_generation(
        TOKENIZER,
        snippet,
        character,
        snippet_role=snippet_role,
        using_system_role=USING_SYSTEM_ROLE,
    )

    generation_module_prompts = [
        prompt_data["prompt"] for prompt_data in all_prompt_data
    ]

    print(f"Running generation on {len(generation_module_prompts)} prompts")

    outputs = model.generate(
        generation_module_prompts, GENERATION_MODULE_SAMPLING_PARAMS
    )

    print(f"Finished generation!")

    all_generation_module_data = []
    for output, prompt_data in zip(outputs, all_prompt_data):
        generated_text = output.outputs[0].text
        generated_text = generated_text.strip()

        new_prompt_data = prompt_data.copy()
        new_prompt_data["response"] = generated_text

        all_generation_module_data.append(new_prompt_data)

    GENERATION_MODULE_OUTPUT_FNAME = args.tempfile_save_dir + f"temp.output.generationmodule.vllm.{GENERATION_MODULE_MODEL_NAME}.dosampleFalse.{GENERATION_MODULE_MAX_TOKENS}maxtok.{GENERATION_MODULE_NUM_BEAMS}beam.jsonl"

    print(
        f"Writing {len(all_generation_module_data)} examples to {GENERATION_MODULE_OUTPUT_FNAME}..."
    )

    # save the generation module output
    with jsonlines.open(GENERATION_MODULE_OUTPUT_FNAME, "w") as writer:
        writer.write_all(all_generation_module_data)

    print(f"FINISHED: saved to {GENERATION_MODULE_OUTPUT_FNAME}")

    #########################################################
    # Simplification
    #########################################################

    SIMPLIFICATION_NUM_BEAMS = args.num_beams

    GENERATION_OUTPUTS = all_generation_module_data

    print(f"Loaded! Found {len(GENERATION_OUTPUTS)} results")

    print(f"Getting sentences...")
    # flatten the data from generation (and get the sentences from each response)
    FLATTENED_GENERATION_OUTPUTS_PER_SENTENCE = flatten_generation_outputs(
        GENERATION_OUTPUTS
    )

    print(
        f"Found {len(FLATTENED_GENERATION_OUTPUTS_PER_SENTENCE)} sentences, starting simplification..."
    )

    #########################################################
    # Simplification
    #########################################################

    print("Getting prompt data")
    # we will save the flattened version of the data, with the entailment information added on top
    all_output_simplified_data = FLATTENED_GENERATION_OUTPUTS_PER_SENTENCE.copy()
    # get prompts for each sentence and this question
    all_prompts_for_this_question = []
    # save sentences so we can get their lengths for generation
    all_sentences = []
    # iterate over per-sentence data
    for sentence_index, query_data in enumerate(
        FLATTENED_GENERATION_OUTPUTS_PER_SENTENCE
    ):
        sentence_of_interest = query_data["sentence_of_interest"]
        all_sentences.append(sentence_of_interest)

        statsandsimps = SIMPLIFICATION_ICL_BASE_QUERY.split("\n\n")[1:]
        statsandsimps = [comb.split("\n") for comb in statsandsimps][:-1]

        # note this is tokenized but both tokenized and non-tokenized should return same result
        cur_prompt = get_chiron_simplification_format_prompt(
            TOKENIZER, statsandsimps, sentence_of_interest, tokenize=True
        )

        all_prompts_for_this_question.append(cur_prompt)

        # update the data with the prompt+question information
        all_output_simplified_data[sentence_index][
            "sentence_from_response_to_split"
        ] = sentence_of_interest

    print(
        f"Running simplification task over model: {len(all_prompts_for_this_question)}"
    )

    # run the simplification task over the dataset, splitting sentences into atomic claims
    output_simplification_texts = (
        run_simplification_task_over_model_with_custom_lengths(
            all_prompts_for_this_question,
            all_sentences,
            model,
            SIMPLIFICATION_NUM_BEAMS,
            TOKENIZER,
        )
    )

    print(
        "Finished running over model, now reformatting our data and filtering bad simplifications..."
    )
    # small step to match the simplification results to the original sentences and save the results in all_output_simplified_data
    reformat_sentences_from_simplification(
        all_output_simplified_data, output_simplification_texts
    )

    print("Reformatting complete")

    SIMPLIFIED_FNAME = args.tempfile_save_dir + f"temp.output.simplified.vllm.{GENERATION_MODULE_MODEL_NAME}.senlen+20maxtok.{SIMPLIFICATION_NUM_BEAMS}beam.jsonl"

    print(
        f"Writing {len(all_output_simplified_data)} examples to {SIMPLIFIED_FNAME}..."
    )

    with jsonlines.open(SIMPLIFIED_FNAME, "w") as writer:
        writer.write_all(all_output_simplified_data)

    print(f"FINISHED: saved to {SIMPLIFIED_FNAME}")

    #########################################################
    # Reasoning
    #########################################################

    # Reasoning parameters
    AMBIGUOUS_NUM_BEAMS = args.reasoning_num_beams
    AMBIGUOUS_MAX_TOKENS = args.reasoning_max_tokens

    INFORMATIVE_NUM_BEAMS = args.reasoning_num_beams
    INFORMATIVE_MAX_TOKENS = args.reasoning_max_tokens

    THOUGHTCHAIN_NUM_BEAMS = args.reasoning_num_beams
    THOUGHTCHAIN_MAX_TOKENS = args.reasoning_max_tokens

    ambiguous_question = get_ambiguous_question()
    ambiguous_icl_question_answers = get_ambiguous_in_context_questions()

    informative_question = get_informative_question()
    informative_icl_question_answers = get_informative_in_context_questions()

    thoughtchain_entailment_questions = get_entailment_questions()

    #########################################################
    # Ambiguity generating
    #########################################################

    SIMPLIFICATION_OUTPUTS = all_output_simplified_data

    print(f"Loaded! Found {len(SIMPLIFICATION_OUTPUTS)} results")

    print(f"Getting sentences...")

    # flatten the data from simplification
    FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE = (
        prepare_simplified_data_for_reasoning(SIMPLIFICATION_OUTPUTS)
    )

    print(
        f"Found {len(FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE)} sentences, starting simplification..."
    )

    print("Getting prompt data")

    # get prompts with the icl questions and simplified data
    all_prompts_for_this_question = get_icl_based_prompts(
        TOKENIZER,
        FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE,
        ambiguous_question,
        ambiguous_icl_question_answers,
        using_system_role=USING_SYSTEM_ROLE,
    )

    print(
        f"Running ambiguous reasoning task over model: {len(all_prompts_for_this_question)}"
    )

    sampling_params = SamplingParams(
        max_tokens=AMBIGUOUS_MAX_TOKENS,
        best_of=AMBIGUOUS_NUM_BEAMS,
        stop=["\n"],
    )

    outputs = model.generate(
        prompt_token_ids=all_prompts_for_this_question, sampling_params=sampling_params
    )

    text_per_prompt = []
    for output in outputs:
        generated_text = output.outputs[0].text
        text_per_prompt.append(generated_text)

    print("Finished running over model, now reformatting and saving...")

    # we will save the flattened version of the data, with the entailment information added on top
    ambiguous_all_data_to_save = FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE.copy()
    for sentence_index, gentext in enumerate(text_per_prompt):
        ambiguous_all_data_to_save[sentence_index][
            "is_ambiguous_response"
        ] = gentext.strip()
        ambiguous_all_data_to_save[sentence_index][
            "is_ambiguous_question"
        ] = get_ambiguous_question()

    print("Reformatting complete")

    AMBIGUOUS_OUTPUT_FNAME = args.tempfile_save_dir + f"temp.output.ambiguous.vllm.{GENERATION_MODULE_MODEL_NAME}.{AMBIGUOUS_MAX_TOKENS}maxtok.{AMBIGUOUS_NUM_BEAMS}beam.jsonl"

    print(
        f"Writing {len(ambiguous_all_data_to_save)} examples to {AMBIGUOUS_OUTPUT_FNAME}..."
    )

    with jsonlines.open(AMBIGUOUS_OUTPUT_FNAME, "w") as writer:
        writer.write_all(ambiguous_all_data_to_save)

    print(f"FINISHED: saved to {AMBIGUOUS_OUTPUT_FNAME}")

    #########################################################
    # Informative generating
    #########################################################

    print("Getting prompt data")

    # get prompts with the icl questions and simplified data
    all_prompts_for_this_question = get_icl_based_prompts(
        TOKENIZER,
        FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE,
        informative_question,
        informative_icl_question_answers,
        using_system_role=USING_SYSTEM_ROLE,
    )

    print(
        f"Running informative reasoning task over model: {len(all_prompts_for_this_question)}"
    )

    sampling_params = SamplingParams(
        max_tokens=INFORMATIVE_MAX_TOKENS,
        best_of=INFORMATIVE_NUM_BEAMS,
        stop=["\n"],
    )

    outputs = model.generate(
        prompt_token_ids=all_prompts_for_this_question, sampling_params=sampling_params
    )

    text_per_prompt = []
    for output in outputs:
        generated_text = output.outputs[0].text
        text_per_prompt.append(generated_text)

    print("Finished running over model, now reformatting and saving...")

    # we will save the flattened version of the data, with the entailment information added on top
    informative_all_data_to_save = FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE.copy()
    for sentence_index, gentext in enumerate(text_per_prompt):
        informative_all_data_to_save[sentence_index][
            "is_informative_response"
        ] = gentext.strip()
        informative_all_data_to_save[sentence_index][
            "is_informative_question"
        ] = get_informative_question()

    print("Reformatting complete")

    INFORMATIVE_OUTPUT_FNAME = args.tempfile_save_dir + f"temp.output.informative.vllm.{GENERATION_MODULE_MODEL_NAME}.{INFORMATIVE_MAX_TOKENS}maxtok.{INFORMATIVE_NUM_BEAMS}beam.jsonl"

    print(
        f"Writing {len(informative_all_data_to_save)} examples to {INFORMATIVE_OUTPUT_FNAME}..."
    )

    with jsonlines.open(INFORMATIVE_OUTPUT_FNAME, "w") as writer:
        writer.write_all(informative_all_data_to_save)

    print(f"FINISHED: saved to {INFORMATIVE_OUTPUT_FNAME}")

    #########################################################
    # Thought chain reasoning
    #########################################################

    all_thoughtchain_data_to_save = FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE.copy()
    prior_answers = []
    # get prompts for each sentence and this question
    all_prompts_for_this_question = []
    # we use chain of reasoning so answer each question for each story
    for entailment_question_index, question in enumerate(
        thoughtchain_entailment_questions
    ):
        print(
            f"ENTAILMENT: {entailment_question_index}/{len(thoughtchain_entailment_questions)}"
        )
        # get prompts for each sentence and this question
        all_prompts_for_this_question = []
        # iterate over per-sentence data
        for sentence_index, query_data in enumerate(
            FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE
        ):
            snippet, sentence_of_interest, character = (
                query_data["snippet"],
                query_data["sentence_of_interest"],
                query_data["character"],
            )

            prior_questions_and_answers = []

            # if there were previous entailment questions, get the prior answers for this sentence
            if entailment_question_index > 0:
                answers = prior_answers[sentence_index]
                existing_entailment_questions = thoughtchain_entailment_questions[
                    :entailment_question_index
                ]
                prior_questions_and_answers = list(
                    zip(existing_entailment_questions, answers)
                )

            # get prompt
            cur_prompt = format_thoughtchain_prompt(
                TOKENIZER,
                character,
                snippet,
                sentence_of_interest,
                question,
                prior_questions_and_answers,
            )

            all_prompts_for_this_question.append(cur_prompt)

            # update the data with the prompt+question information
            all_thoughtchain_data_to_save[sentence_index][
                f"entailment_prompt_{entailment_question_index}"
            ] = cur_prompt
            all_thoughtchain_data_to_save[sentence_index][
                f"entailment_question_{entailment_question_index}"
            ] = question
            all_thoughtchain_data_to_save[sentence_index][
                "entailment_question_index"
            ] = entailment_question_index
            all_thoughtchain_data_to_save[sentence_index][
                "prior_entailment_qa"
            ] = prior_questions_and_answers

        sampling_params = SamplingParams(
            max_tokens=THOUGHTCHAIN_MAX_TOKENS,
            best_of=THOUGHTCHAIN_NUM_BEAMS,
            stop=["\n"],
        )
        # note we prepend the answer: so this isn't in tokens
        outputs = model.generate(all_prompts_for_this_question, sampling_params)
        text_per_prompt = []
        for output in outputs:
            generated_text = output.outputs[0].text
            text_per_prompt.append(generated_text)

        print(f"len(text_per_prompt): {len(text_per_prompt)}")
        # is this the first set of answers?
        if len(prior_answers) == 0:
            for t in text_per_prompt:
                nice_response = t.strip()
                nice_response = regexspace.sub(" ", nice_response)
                nice_response = regexlines.sub(" ", nice_response)

                prior_answers.append([nice_response])
        else:
            # add each response to the corresponding sentence index (so we have the saved answers for each question)
            for i, t in enumerate(text_per_prompt):
                # reformat response to be nice and add to the prior answers
                nice_response = t.strip()
                nice_response = regexspace.sub(" ", nice_response)
                nice_response = regexlines.sub(" ", nice_response)
                prior_answers[i].append(nice_response)

        for sentence_index, gentext in enumerate(text_per_prompt):
            all_thoughtchain_data_to_save[sentence_index][
                f"entailment_response_{entailment_question_index}"
            ] = gentext.strip()

    print("Finished running over model, now saving...")

    THOUGHTCHAIN_OUTPUT_FNAME = args.tempfile_save_dir + f"temp.output.thoughtchain.{GENERATION_MODULE_MODEL_NAME}.{THOUGHTCHAIN_MAX_TOKENS}maxtok.{THOUGHTCHAIN_NUM_BEAMS}beam.jsonl"

    print(
        f"Writing {len(all_thoughtchain_data_to_save)} examples to {THOUGHTCHAIN_OUTPUT_FNAME}..."
    )

    with jsonlines.open(THOUGHTCHAIN_OUTPUT_FNAME, "w") as writer:
        writer.write_all(all_thoughtchain_data_to_save)

    print(f"FINISHED: saved to {THOUGHTCHAIN_OUTPUT_FNAME}")

    #########################################################
    # Setup Verification Classifier
    #########################################################
    TOKENIZER = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    print(f"Tokenizer loaded!")

    possible_snippet_character_statements = set()
    dataset_mappings = {"chain": {}, "amb": {}, "inf": {}}
    for dataset, name in zip(
        [
            all_thoughtchain_data_to_save,
            ambiguous_all_data_to_save,
            informative_all_data_to_save,
        ],
        ["chain", "amb", "inf"],
    ):
        for point in dataset:
            snippet = point["snippet"]
            character = point["character"]
            statement = point["sentence_of_interest"]
            scs = (snippet, character, statement)
            possible_snippet_character_statements.add(scs)
            dataset_mappings[name][scs] = point
        print("loading", name, len(dataset), "->", len(dataset_mappings[name]))
        print(
            f"total snippet-character-statements so far: {len(possible_snippet_character_statements)}"
        )

    temp_combined_data = get_combined_data(
        GENERATION_MODULE_CHECKPOINT_NAME,
        TOKENIZER,
        dataset_mappings,
        possible_snippet_character_statements,
        input_chain_data=all_thoughtchain_data_to_save,
        input_inf_data=informative_all_data_to_save,
        input_amb_data=ambiguous_all_data_to_save,
    )

    combined_df = pd.DataFrame(temp_combined_data, columns=["text"])

    COMBINED_OUTPUT_FNAME = args.tempfile_save_dir + f"temp_combined_datatoannotate_{GENERATION_MODULE_MODEL_NAME}.csv"

    combined_df.to_csv(COMBINED_OUTPUT_FNAME)
    print(f"Wrote {COMBINED_OUTPUT_FNAME}")

    #########################################################
    # Verification Classifier
    #########################################################
    del model

    model = AutoModelForCausalLM.from_pretrained(
        args.verification_model,
        device_map="auto",
        pad_token_id=TOKENIZER.eos_token_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    print("LOADING DATASET")

    combined_df = pd.read_csv(COMBINED_OUTPUT_FNAME)

    print("Running model over combined")
    combined_responses = get_model_predictions_across_dataset(
        model, TOKENIZER, combined_df
    )

    VERIFICATION_OUTPUT_FNAME = args.tempfile_save_dir + f"combined_output_eval_results_{GENERATION_MODULE_MODEL_NAME}.csv"
    print(f"Saving combined to: {VERIFICATION_OUTPUT_FNAME}")

    combined_responses.to_csv(VERIFICATION_OUTPUT_FNAME)

    OPTIMIZED_FNAME = args.tempfile_save_dir + f"temp.output.memory_efficient.vllm.{GENERATION_MODULE_MODEL_NAME}.{SIMPLIFICATION_NUM_BEAMS}beam.jsonl"
    VERIFICATION_OUTPUT_FNAME = (
        args.tempfile_save_dir + f"combined_output_eval_results_{GENERATION_MODULE_MODEL_NAME}.csv"
    )

    with jsonlines.open(SIMPLIFIED_FNAME, "r") as reader:
        simplified_output_predictions = list(reader)
    with jsonlines.open(GENERATION_MODULE_OUTPUT_FNAME, "r") as reader:
        original_output_predictions = list(reader)

    optimized_output_predictions = fast_new_output_predictions_from_simpified_and_original_outputs(
        simplified_output_predictions,
        original_output_predictions,
    )

    print(f"Writing combined output predictions to {OPTIMIZED_FNAME}")
    with jsonlines.open(OPTIMIZED_FNAME, "w") as writer:
        writer.write_all(optimized_output_predictions)

    del simplified_output_predictions
    del original_output_predictions
    del optimized_output_predictions

    get_character_sheet_for_one_snippet_data(
        OPTIMIZED_FNAME, VERIFICATION_OUTPUT_FNAME, args.output_file
    )


if __name__ == "__main__":
    main()
