def main():
    import argparse
    import torch
    import os
    from chiron_dataset_setup_utils import get_combined_data
    from chiron_filter_utils import get_model_predictions_across_dataset
    from chiron_generation_module_utils import get_prompt_data_for_chiron_generation

    import jsonlines
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from chiron_simplification_utils import (
        SIMPLIFICATION_ICL_BASE_QUERY,
        flatten_generation_outputs,
        get_chiron_simplification_format_prompt,
        reformat_sentences_from_simplification,
        run_simplification_task_over_model_with_custom_lengths,
    )

    import pandas as pd
    import pickle
    
    import re
    regexspace = re.compile(r"\ +", re.IGNORECASE)
    regexlines = re.compile(r"(\n(\ )?)+", re.IGNORECASE)

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
        make_dataset_write_and_get_stats,
    )

    parser = argparse.ArgumentParser(
        description="Generate character sheets from provided jsonl dataset"
    )
    parser.add_argument(
        "--stories-fname",
        type=str,
        default="dummy_example.jsonl",
        help="Name of the file containing the stories to process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output charcter sheet files to (default: None, will save to dataset-name/)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=torch.cuda.device_count(),
        help="Number of GPUs to use for vllm tensor parallelism (default: number of available CUDA devices)",
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        default=-1,
        help="Maximum number of stories to process, if -1, will process all",
    )
    parser.add_argument(
        "--max-snippets-per-story",
        type=int,
        default=-1,
        help="Maximum number of snippets to process per story, if -1, will process all",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="dummy_example",
        help="Name of the dataset to process",
    )
    parser.add_argument(
        "--tempfile-save-dir",
        type=str,
        default="tmp/",
        help="Directory to save temporary files to (default: tmp/). Will create if it does not exist.",
    )
    parser.add_argument(
        "--dont-get-statistics",
        action="store_false",
        help="Get statistics (e.g. density) for the character sheets and underlying stories, can be slow. Defaults to True",
    )
    parser.add_argument(
        "--skip-to-verification",
        action="store_true",
        help="Skip to verification step, useful when you have already generated data and reasoning steps and just want to run the classifier (saves time loading generation model). Defaults to False.",
    )

    # Generation parameters
    parser.add_argument(
        "--generation-checkpoint-name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HuggingFace model checkpoint (default: mistralai/Mistral-7B-Instruct-v0.2). Paper uses Mistral-7B-Instruct-v0.2 for downstream tasks, but other models are supported (tested with llama 3.1).",
    )
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

    args = parser.parse_args()

    # create the tempfile save dir if it does not exist
    if not os.path.exists(args.tempfile_save_dir):
        os.makedirs(args.tempfile_save_dir)

    max_snippets_per_story = args.max_snippets_per_story

    SAVING_NAME = args.dataset_name
    OUTPUT_CHARACTER_SHEET_DIR = args.output_dir if args.output_dir is not None else SAVING_NAME

    SKIP_TO_VERIFICATION = args.skip_to_verification

    ORIGINAL_STORIES_INPUT_FNAME = args.stories_fname

    ##########################################################################
    # Generation Module
    ##########################################################################
    GENERATION_MODULE_MAX_TOKENS = args.max_tokens
    GENERATION_MODULE_NUM_BEAMS = args.num_beams

    GENERATION_MODULE_CHECKPOINT_NAME = args.generation_checkpoint_name
    GENERATION_MODULE_MODEL_NAME = GENERATION_MODULE_CHECKPOINT_NAME.split("/")[
        -1
    ].lower()
    USING_SYSTEM_ROLE = "llama" in GENERATION_MODULE_CHECKPOINT_NAME.lower()

    model = None
    TOKENIZER = None
    if not SKIP_TO_VERIFICATION:
        print(
            f"Loading generation model and tokenizer: {GENERATION_MODULE_CHECKPOINT_NAME}"
        )
        model = LLM(
            model=GENERATION_MODULE_CHECKPOINT_NAME,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        print(f"Model loaded!")
        TOKENIZER = AutoTokenizer.from_pretrained(GENERATION_MODULE_CHECKPOINT_NAME)
        TOKENIZER.padding_side = "right"
        TOKENIZER.add_special_tokens({"pad_token": "[PAD]"})
        print(f"Tokenizer loaded!")

    GENERATION_OUTPUT_FNAME = (
        args.tempfile_save_dir
        + f"temp.{SAVING_NAME}.output.generationmodule.vllm.{GENERATION_MODULE_MODEL_NAME}.dosampleFalse.{args.max_tokens}maxtok.{args.num_beams}beam.jsonl"
    )

    def generation(fname=None):
        # if fname is not None, we will load the data from the fname instead of generating it
        if fname is not None and os.path.exists(fname):
            with jsonlines.open(fname, "r") as reader:
                GENERATION_OUTPUTS = list(reader)

            print(f"Getting sentences...")
            FLATTENED_GENERATION_OUTPUTS_PER_SENTENCE = flatten_generation_outputs(
                GENERATION_OUTPUTS
            )

            print(
                f"Found {len(FLATTENED_GENERATION_OUTPUTS_PER_SENTENCE)} sentences, starting simplification..."
            )
            return FLATTENED_GENERATION_OUTPUTS_PER_SENTENCE

        # we have not already generated the data
        print(f"Loading input stories from {ORIGINAL_STORIES_INPUT_FNAME}...")
        with jsonlines.open(ORIGINAL_STORIES_INPUT_FNAME) as reader:
            STORIES_TO_BASE_CHARACTER_SHEETS_ON = list(reader)

        print(f"Loaded! Found {len(STORIES_TO_BASE_CHARACTER_SHEETS_ON)} stories")

        if args.max_stories > 0:
            STORIES_TO_BASE_CHARACTER_SHEETS_ON = STORIES_TO_BASE_CHARACTER_SHEETS_ON[
                : args.max_stories
            ]
        unique_stories = set(
            [x["story_id"] for x in STORIES_TO_BASE_CHARACTER_SHEETS_ON]
        )
        unique_storychars = set(
            [
                (x["story_id"], x["character"])
                for x in STORIES_TO_BASE_CHARACTER_SHEETS_ON
            ]
        )
        print(
            f"Generating with {len(unique_stories)} unique stories and {len(unique_storychars)} unique story-character pairs out of {len(STORIES_TO_BASE_CHARACTER_SHEETS_ON)}"
        )

        GENERATION_MODULE_SAMPLING_PARAMS = SamplingParams(
            max_tokens=GENERATION_MODULE_MAX_TOKENS,
            # use_beam_search=True,
            # temperature=0,
            best_of=GENERATION_MODULE_NUM_BEAMS,
            stop=["\n"],
        )

        all_generation_module_data_pre_generation = []
        generation_module_prompts = []
        # for every story
        for storychar in STORIES_TO_BASE_CHARACTER_SHEETS_ON:
            all_snippets = storychar["snippets"]

            # for every snippet in the story, get the prompt and data
            for index, snippet_data in enumerate(all_snippets):
                if index > 0 and index > max_snippets_per_story:
                    break
                # some datasets have roles for snippets (the character perspective of the snippet), some don't
                snippet_role = None
                if len(snippet_data) == 3:
                    snippet_role, original_text, snippet = snippet_data
                else:
                    original_text, snippet = snippet_data

                all_prompt_data = get_prompt_data_for_chiron_generation(
                    TOKENIZER,
                    snippet,
                    storychar["character"],
                    snippet_role=snippet_role,
                    using_system_role=USING_SYSTEM_ROLE,
                )
                for prompt_data in all_prompt_data:
                    prompt_data["story_id"] = storychar["story_id"]

                    generation_module_prompts.append(prompt_data["prompt"])
                    all_generation_module_data_pre_generation.append(prompt_data)

        print(f"Running generation on {len(generation_module_prompts)} prompts")
        unique_stories = set(
            [x["story_id"] for x in all_generation_module_data_pre_generation]
        )
        unique_storychars = set(
            [
                (x["story_id"], x["character"])
                for x in all_generation_module_data_pre_generation
            ]
        )
        print(
            f"Unique stories: {len(unique_stories)} and unique story-character pairs: {len(unique_storychars)}"
        )

        outputs = model.generate(
            generation_module_prompts, GENERATION_MODULE_SAMPLING_PARAMS
        )

        print(f"Finished generation!")

        all_generation_module_data = []
        for output, prompt_data in zip(
            outputs, all_generation_module_data_pre_generation
        ):
            generated_text = output.outputs[0].text
            generated_text = generated_text.strip()

            new_prompt_data = prompt_data.copy()
            new_prompt_data["response"] = generated_text

            all_generation_module_data.append(new_prompt_data)

        print(f"Writing {len(all_generation_module_data)} examples to {fname}...")

        with jsonlines.open(fname, "w") as writer:
            writer.write_all(all_generation_module_data)

        print(f"FINISHED: saved to {fname}")

        del all_generation_module_data_pre_generation
        del generation_module_prompts
        del GENERATION_MODULE_SAMPLING_PARAMS

        GENERATION_OUTPUTS = all_generation_module_data

        print(f"Loaded! Found {len(GENERATION_OUTPUTS)} results")

        print(f"Getting sentences...")
        FLATTENED_GENERATION_OUTPUTS_PER_SENTENCE = flatten_generation_outputs(
            GENERATION_OUTPUTS
        )

        print(
            f"Found {len(FLATTENED_GENERATION_OUTPUTS_PER_SENTENCE)} sentences, starting simplification..."
        )
        return FLATTENED_GENERATION_OUTPUTS_PER_SENTENCE

    if not SKIP_TO_VERIFICATION:
        FLATTENED_GENERATION_OUTPUTS_PER_SENTENCE = generation(
            fname=GENERATION_OUTPUT_FNAME
        )

    #########################################################
    # Simplification
    #########################################################
    SIMPLIFICATION_NUM_BEAMS = args.num_beams
    SIMPLIFICATION_OUTPUT_FNAME = (
        args.tempfile_save_dir
        + f"temp.{SAVING_NAME}.output.simplified.vllm.{GENERATION_MODULE_MODEL_NAME}.{SIMPLIFICATION_NUM_BEAMS}beam.jsonl"
    )

    def simplification(
        FLATTENED_GENERATION_OUTPUTS_PER_SENTENCE, model, TOKENIZER, fname=None
    ):
        if fname is not None and os.path.exists(fname):
            with jsonlines.open(fname) as reader:
                all_output_simplified_data = list(reader)
            print(
                f"Simplification: loaded {len(all_output_simplified_data)} examples from {fname}"
            )
            return all_output_simplified_data

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

            # note this is tokenized but both should return same result

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

        reformat_sentences_from_simplification(
            all_output_simplified_data, output_simplification_texts
        )

        print("Reformatting complete")
        print(f"Writing {len(all_output_simplified_data)} examples to {fname}...")

        with jsonlines.open(fname, "w") as writer:
            writer.write_all(all_output_simplified_data)

        print(f"FINISHED: saved to {fname}")

        del all_prompts_for_this_question
        del all_sentences
        return all_output_simplified_data

    if not SKIP_TO_VERIFICATION:
        simplification(
            FLATTENED_GENERATION_OUTPUTS_PER_SENTENCE,
            model,
            TOKENIZER,
            fname=SIMPLIFICATION_OUTPUT_FNAME,
        )

    with jsonlines.open(SIMPLIFICATION_OUTPUT_FNAME, "r") as reader:
        SIMPLIFICATION_OUTPUTS = list(reader)

    unique_story_ids = set([x["story_id"] for x in SIMPLIFICATION_OUTPUTS])
    num_unique_stories = len(unique_story_ids)
    num_unique_storychars = len(
        set([(x["story_id"], x["character"]) for x in SIMPLIFICATION_OUTPUTS])
    )
    print(
        f"Loaded! Found {len(SIMPLIFICATION_OUTPUTS)} results with {num_unique_stories} unique stories and {num_unique_storychars} unique story-character pairs"
    )
    print(f"unique stories: {unique_story_ids}")

    #########################################################
    # Reasoning
    #########################################################
    AMBIGUOUS_NUM_BEAMS = args.reasoning_num_beams
    AMBIGUOUS_MAX_TOKENS = args.reasoning_max_tokens

    ambiguous_question = get_ambiguous_question()
    ambiguous_icl_question_answers = get_ambiguous_in_context_questions()

    INFORMATIVE_NUM_BEAMS = args.reasoning_num_beams
    INFORMATIVE_MAX_TOKENS = args.reasoning_max_tokens

    informative_question = get_informative_question()
    informative_icl_question_answers = get_informative_in_context_questions()

    THOUGHTCHAIN_NUM_BEAMS = args.reasoning_num_beams
    THOUGHTCHAIN_MAX_TOKENS = args.reasoning_max_tokens
    thoughtchain_entailment_questions = get_entailment_questions()

    print(f"Getting sentences...")

    # flatten the data from simplification
    FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE_FNAME = (
        args.tempfile_save_dir + f"temp.{SAVING_NAME}.output.flattened.jsonl"
    )
    if not os.path.exists(FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE_FNAME):
        FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE = (
            prepare_simplified_data_for_reasoning(SIMPLIFICATION_OUTPUTS)
        )

        with jsonlines.open(
            FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE_FNAME, "w"
        ) as writer:
            writer.write_all(FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE)
    else:
        with jsonlines.open(
            FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE_FNAME, "r"
        ) as reader:
            FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE = list(reader)

    del SIMPLIFICATION_OUTPUTS

    #########################################################
    # Ambiguity generating
    #########################################################
    AMBIGUOUS_OUTPUT_FNAME = (
        args.tempfile_save_dir
        + f"temp.{SAVING_NAME}.output.ambiguous.vllm.{GENERATION_MODULE_MODEL_NAME}.{AMBIGUOUS_MAX_TOKENS}maxtok.{AMBIGUOUS_NUM_BEAMS}beam.jsonl"
    )

    def ambiguity(fname=None):
        if fname is not None and os.path.exists(fname):
            with jsonlines.open(fname) as reader:
                ambiguous_all_data_to_save = list(reader)
            print(
                f"Ambiguity: loaded {len(ambiguous_all_data_to_save)} examples from {fname}"
            )
            return ambiguous_all_data_to_save

        print("Working on Ambiguity")

        print(
            f"Found {len(FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE)} sentences, starting ambiguity..."
        )

        print("Getting prompt data")

        if os.path.exists(args.tempfile_save_dir + "ambiguous_prompts.pkl"):
            with open(args.tempfile_save_dir + "ambiguous_prompts.pkl", "rb") as f:
                all_prompts_for_this_question = pickle.load(f)
        else:
            # get prompts with the icl questions and simplified data
            all_prompts_for_this_question = get_icl_based_prompts(
                TOKENIZER,
                FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE,
                ambiguous_question,
                ambiguous_icl_question_answers,
                using_system_role=USING_SYSTEM_ROLE,
            )

            # this can take a while, so save the prompts
            with open(args.tempfile_save_dir + "ambiguous_prompts.pkl", "wb") as f:
                pickle.dump(all_prompts_for_this_question, f)

        print(
            f"Running ambiguous reasoning task over model: {len(all_prompts_for_this_question)}"
        )

        sampling_params = SamplingParams(
            max_tokens=AMBIGUOUS_MAX_TOKENS,
            # use_beam_search=True,
            # temperature=0,
            best_of=AMBIGUOUS_NUM_BEAMS,
            stop=["\n"],
        )

        outputs = model.generate(
            prompt_token_ids=all_prompts_for_this_question,
            sampling_params=sampling_params,
        )

        text_per_prompt = []
        for output in outputs:
            generated_text = output.outputs[0].text
            text_per_prompt.append(generated_text)

        print("Finished running over model, now reformatting and saving...")

        # we will save the flattened version of the data, with the entailment information added on top
        ambiguous_all_data_to_save = (
            FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE.copy()
        )
        for sentence_index, gentext in enumerate(text_per_prompt):
            ambiguous_all_data_to_save[sentence_index][
                "is_ambiguous_response"
            ] = gentext.strip()
            ambiguous_all_data_to_save[sentence_index][
                "is_ambiguous_question"
            ] = get_ambiguous_question()

        print("Reformatting complete")

        print(f"Writing {len(ambiguous_all_data_to_save)} examples to {fname}...")

        with jsonlines.open(fname, "w") as writer:
            writer.write_all(ambiguous_all_data_to_save)

        print(f"FINISHED: saved to {fname}")

        del all_prompts_for_this_question
        del sampling_params
        return ambiguous_all_data_to_save

    ambiguous_all_data_to_save = ambiguity(fname=AMBIGUOUS_OUTPUT_FNAME)

    #########################################################
    # Informative generating
    #########################################################
    INFORMATIVE_OUTPUT_FNAME = (
        args.tempfile_save_dir
        + f"temp.{SAVING_NAME}.output.informative.vllm.{GENERATION_MODULE_MODEL_NAME}.{INFORMATIVE_MAX_TOKENS}maxtok.{INFORMATIVE_NUM_BEAMS}beam.jsonl"
    )

    def informative(fname=None):
        if fname is not None and os.path.exists(fname):
            with jsonlines.open(fname) as reader:
                informative_all_data_to_save = list(reader)
            print(
                f"Informative: loaded {len(informative_all_data_to_save)} examples from {fname}"
            )
            return informative_all_data_to_save
        print("Working on Informative")

        print("Getting prompt data")

        if os.path.exists(args.tempfile_save_dir + "informative_prompts.pkl"):
            with open(args.tempfile_save_dir + "informative_prompts.pkl", "rb") as f:
                all_prompts_for_this_question = pickle.load(f)
        else:
            # get prompts with the icl questions and simplified data
            all_prompts_for_this_question = get_icl_based_prompts(
                TOKENIZER,
                FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE,
                informative_question,
                informative_icl_question_answers,
                using_system_role=USING_SYSTEM_ROLE,
            )

            # this can take a while, so save the prompts
            with open(args.tempfile_save_dir + "informative_prompts.pkl", "wb") as f:
                pickle.dump(all_prompts_for_this_question, f)

        print(
            f"Running informative reasoning task over model: {len(all_prompts_for_this_question)}"
        )

        sampling_params = SamplingParams(
            max_tokens=INFORMATIVE_MAX_TOKENS,
            # use_beam_search=True,
            # temperature=0,
            best_of=INFORMATIVE_NUM_BEAMS,
            stop=["\n"],
        )

        outputs = model.generate(
            prompt_token_ids=all_prompts_for_this_question,
            sampling_params=sampling_params,
        )

        text_per_prompt = []
        for output in outputs:
            generated_text = output.outputs[0].text
            text_per_prompt.append(generated_text)

        print("Finished running over model, now reformatting and saving...")

        # we will save the flattened version of the data, with the entailment information added on top
        informative_all_data_to_save = (
            FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE.copy()
        )
        for sentence_index, gentext in enumerate(text_per_prompt):
            informative_all_data_to_save[sentence_index][
                "is_informative_response"
            ] = gentext.strip()
            informative_all_data_to_save[sentence_index][
                "is_informative_question"
            ] = get_informative_question()

        print("Reformatting complete")

        print(f"Writing {len(informative_all_data_to_save)} examples to {fname}...")

        with jsonlines.open(fname, "w") as writer:
            writer.write_all(informative_all_data_to_save)

        print(f"FINISHED: saved to {fname}")

        del all_prompts_for_this_question
        del sampling_params
        return informative_all_data_to_save

    # informative_all_data_to_save = informative()
    informative_all_data_to_save = informative(fname=INFORMATIVE_OUTPUT_FNAME)

    #########################################################
    # Thought chain reasoning
    #########################################################
    THOUGHTCHAIN_OUTPUT_FNAME = (
        args.tempfile_save_dir
        + f"temp.{SAVING_NAME}.output.thoughtchain.vllm.{GENERATION_MODULE_MODEL_NAME}.{THOUGHTCHAIN_MAX_TOKENS}maxtok.{THOUGHTCHAIN_NUM_BEAMS}beam.jsonl"
    )

    def thought_chain(fname=None):
        if fname is not None and os.path.exists(fname):
            with jsonlines.open(fname) as reader:
                all_thoughtchain_data_to_save = list(reader)
            print(
                f"Thought Chain: loaded {len(all_thoughtchain_data_to_save)} examples from {fname}"
            )
            return all_thoughtchain_data_to_save

        print("Working on Thought Chain")

        all_thoughtchain_data_to_save = (
            FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE.copy()
        )
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
                # use_beam_search=True,
                # temperature=0,
                best_of=THOUGHTCHAIN_NUM_BEAMS,
                stop=["\n"],
            )
            # note we prepend the answer: so this isn't in tokens
            outputs = model.generate(
                all_prompts_for_this_question, sampling_params, use_tqdm=False
            )
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

        print(f"Writing {len(all_thoughtchain_data_to_save)} examples to {fname}...")

        with jsonlines.open(fname, "w") as writer:
            writer.write_all(all_thoughtchain_data_to_save)

        print(f"FINISHED: saved to {fname}")

        del all_prompts_for_this_question
        del sampling_params
        return all_thoughtchain_data_to_save

    # all_thoughtchain_data_to_save = thought_chain()
    all_thoughtchain_data_to_save = thought_chain(THOUGHTCHAIN_OUTPUT_FNAME)

    #########################################################
    # Setup Verification Classifier
    #########################################################
    print("Setting up verification classifier")
    del model
    del FLATTENED_SIMPLIFICATION_OUTPUTS_PER_SENTENCE
    # we have to use a different tokenizer for this specifically
    del TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    print(f"Tokenizer loaded!")

    COMBINED_OUTPUT_FNAME = (
        args.tempfile_save_dir
        + f"temp_combined_datatoannotate_{GENERATION_MODULE_MODEL_NAME}.csv"
    )

    def combine_datasets(fname=None):
        if fname is not None and os.path.exists(fname):
            combined_df = pd.read_csv(fname)
            print(f"Combined: loaded {len(combined_df)} examples from {fname}")
            return combined_df

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
            print(name, len(dataset), "->", len(dataset_mappings[name]))
            print(f"total so far: {len(possible_snippet_character_statements)}")

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
        combined_df.to_csv(fname)
        print(f"Wrote {fname}")
        return combined_df

    combined_df = combine_datasets(fname=COMBINED_OUTPUT_FNAME)

    del all_thoughtchain_data_to_save
    del ambiguous_all_data_to_save
    del informative_all_data_to_save

    #########################################################
    # Verification Classifier
    #########################################################
    print("Verifying")

    model = AutoModelForCausalLM.from_pretrained(
        "agurung/character_sheet_entailment_model___mistral7bv2",
        device_map="auto",
        pad_token_id=TOKENIZER.eos_token_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    VERIFICATION_OUTPUT_FNAME = (
        args.tempfile_save_dir
        + f"combined_output_eval_results_{SAVING_NAME}_{GENERATION_MODULE_MODEL_NAME}.csv"
    )

    def run_verification_classifier(fname=None):
        if fname is not None and os.path.exists(fname):
            verified_results = pd.read_csv(fname)
            print(f"Combined: loaded {len(verified_results)} examples from {fname}")
            return verified_results

        print("Running model over combined")
        verified_results = get_model_predictions_across_dataset(
            model, TOKENIZER, combined_df
        )

        print(f"Saving combined to: {fname}")

        verified_results.to_csv(fname)
        return verified_results

    run_verification_classifier(fname=VERIFICATION_OUTPUT_FNAME)

    OPTIMIZED_FNAME = (
        args.tempfile_save_dir
        + f"temp.{SAVING_NAME}.output.memory_efficient.vllm.{GENERATION_MODULE_MODEL_NAME}.{SIMPLIFICATION_NUM_BEAMS}beam.jsonl"
    )

    with jsonlines.open(SIMPLIFICATION_OUTPUT_FNAME, "r") as reader:
        simplified_output_predictions = list(reader)
    with jsonlines.open(GENERATION_OUTPUT_FNAME, "r") as reader:
        original_output_predictions = list(reader)

    def get_optimized_predictions(fname=None):
        # reformat the predictions from the generation module into a more memory efficient format
        # this is useful for putting all of the character sheets together, since we have to
        # load a lot of data at once
        if fname is not None and os.path.exists(fname):
            with jsonlines.open(fname, "r") as reader:
                optimized_output_predictions = list(reader)
            print(
                f"Optimized: loaded {len(optimized_output_predictions)} examples from {fname}"
            )
            return optimized_output_predictions

        optimized_output_predictions = (
            fast_new_output_predictions_from_simpified_and_original_outputs(
                simplified_output_predictions, original_output_predictions
            )
        )

        with jsonlines.open(fname, "w") as writer:
            writer.write_all(optimized_output_predictions)
        print(
            f"Optimized: saved {len(optimized_output_predictions)} examples to {fname}"
        )
        return optimized_output_predictions

    get_optimized_predictions(fname=OPTIMIZED_FNAME)

    make_dataset_write_and_get_stats(
        output_predictions_fname=OPTIMIZED_FNAME,
        input_original_data_fname=ORIGINAL_STORIES_INPUT_FNAME,
        all_preds_fname=VERIFICATION_OUTPUT_FNAME,
        DATASET_NAME=SAVING_NAME,
        FOLDER_NAME=OUTPUT_CHARACTER_SHEET_DIR,
        get_statistics=args.dont_get_statistics,
        reform_for_presnippet=False,
        FILTER_TO_ROLE=False,
    )


if __name__ == "__main__":
    main()
