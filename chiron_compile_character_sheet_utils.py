from tqdm import tqdm
import jsonlines

import os
import shutil
import pandas as pd
import itertools


def flatten_list(nested_list):
    # simple helper function to flatten a nested list
    return list(itertools.chain(*nested_list))


from chiron_utils import get_sentences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from typing import List, Dict, Any, Set, Tuple, Optional, Callable

import re

regexspace = re.compile(r"\ +", re.IGNORECASE)
regexlines = re.compile(r"(\n(\ )?)+", re.IGNORECASE)


def fast_new_output_predictions_from_simpified_and_original_outputs(
    simplified_output_predictions: List[Dict[str, Any]],
    original_output_predictions: List[List[Dict[str, Any]]],
) -> List[List[Dict[str, Any]]]:
    """Takes simplified and original outputs and aligns them. Output has same shape as original outputs, but with simplified responses.
    Worth noting that if you have a really large dataset, this will take a lot of memory since it stores a lot of intermediate data in a dictionary for quick lookups.
    If you want to save memory, you can just iterate over the original outputs and do the lookup yourself.

    :param simplified_output_predictions: A list of dictionaries containing simplified predictions.
    :param original_output_predictions: A list of lists of dictionaries containing original predictions.
    :return: A list of lists of dictionaries containing the new datapoints with simplified responses.
    """
    new_output_predictions = []
    # iterate over simplified outputs and group them by snippet, character, question
    # scq (snippet, character, question) -> list of outputs
    scq_to_relevant_outputs = {}
    for output in simplified_output_predictions:
        cur_scq = tuple([output[c] for c in ["snippet", "character", "question"]])
        if cur_scq not in scq_to_relevant_outputs:
            # haven't seen this combination of snippet, character, question before, so initialize an empty list
            scq_to_relevant_outputs[cur_scq] = []
        scq_to_relevant_outputs[cur_scq].append(output)

    # original_output_predictions is a flattened list of all of the original outputs, want to group them by story_id and character
    story_id_and_character_to_original_outputs = {}
    for output in original_output_predictions:
        cur_story_id = output["story_id"]
        cur_character = output["character"]
        if (cur_story_id, cur_character) not in story_id_and_character_to_original_outputs:
            story_id_and_character_to_original_outputs[(cur_story_id, cur_character)] = []
        story_id_and_character_to_original_outputs[(cur_story_id, cur_character)].append(output)
    
    reorganized_original_output_predictions = list(story_id_and_character_to_original_outputs.values())
    
    del story_id_and_character_to_original_outputs # free up memory

    # iterate over original outputs and align them with simplified outputs
    for cur_story in tqdm(reorganized_original_output_predictions):
        new_story_output_snippets = []
        for snip in cur_story:
            new_output_snippet = {
                "snippet": snip["snippet"],
                "character": snip["character"],
                "question": snip["question"],
                "story_id": snip["story_id"],
            }

            snip_char_ques = tuple(
                [snip[c] for c in ["snippet", "character", "question"]]
            )

            relevant_outputs_from_simplified = scq_to_relevant_outputs[snip_char_ques]

            # make it a list instead of raw text since we're going through and filtering anyway
            new_output_response = []
            for r in relevant_outputs_from_simplified:
                simplified_response = r["response"]
                sentences = get_sentences(simplified_response)
                new_output_response.extend(sentences)

            new_output_snippet["response"] = new_output_response

            new_story_output_snippets.append(new_output_snippet)
        new_output_predictions.append(new_story_output_snippets)
    print(f"len(new_output_predictions): {len(new_output_predictions)}")
    return new_output_predictions


# some helper functions for getting the snippet, statement, and character from a text
# note, only works because we have a very specific format
to_snippet = lambda text: text.split("Story Section:\n")[1].split(
    "\n\nPlease answer the following que"
)[0]
to_statement = lambda text: text.split("\n\nStatement: ")[1].split("\n\nQuestion: Is")[
    0
]
to_character = lambda text: text.split(
    "You are a helpful and expert writing assistant. You will be given a section of a story or screenplay from the perspective of "
)[1].split(". Please")[0]


def display_text_to_statement_snippet_char(text: str) -> Tuple[str, str, str]:
    """Extracts the statement, snippet, and character from a given text.

    :param str text: The input text containing the statement, snippet, and character information.
    :return Tuple[str, str, str]: A tuple containing (statement, snippet, character).
    """
    statement = to_statement(text)
    snippet = to_snippet(text)
    character = to_character(text)

    return statement, snippet, character


def get_dialogue_qs() -> List[str]:
    q1 = f"What, if anything, have we learned about how this character speaks from this snippet?"
    qs = [q1]
    return qs


def get_physical_qs() -> List[str]:
    q1 = f"What, if any, physical descriptions of this character are in this snippet?"
    q2 = f"What, if any, descriptions of this character's personality are in this snippet?"
    qs = [q1, q2]
    return qs


def get_knowledge_qs() -> List[str]:
    q1 = f"What, if any, factual information is given about this character in this snippet?"
    q2 = f"What, if any, information has this character learned in this snippet?"
    qs = [q1, q2]
    return qs


def get_plot_qs() -> List[str]:
    q1 = f"What, if any, goals does this character gain in this snippet that they wish to accomplish in the future?"
    q2 = f"What, if any, goals does this character complete in this snippet?"
    q3 = f"How, if at all, does this character's internal motivations change in this snippet?"

    qs = [q1, q2, q3]
    return qs


# helper snippet to get the type of question
question_to_type = {}
for d in get_dialogue_qs():
    question_to_type[d] = "dialogue"

for d in get_physical_qs():
    question_to_type[d] = "physical"

for d in get_knowledge_qs():
    question_to_type[d] = "knowledge"

for d in get_plot_qs():
    question_to_type[d] = "plot"

for d in [
    "Summarize everything we have learned about this character in this snippet. Include aspects of the character like how they speak, what they look like, their personality, their goals, etc."
]:
    question_to_type[d] = "summarize_snippet"

for d in [
    "Summarize everything we have learned about this character across these snippets. Include aspects of the character like how they speak, what they look like, their personality, their goals, etc."
]:
    question_to_type[d] = "summarize_story"

# list all questions out so we can iterate over them
unique_questions = [
    "How, if at all, does this character's internal motivations change in this snippet?",
    "What, if any, descriptions of this character's personality are in this snippet?",
    "What, if any, factual information is given about this character in this snippet?",
    "What, if any, goals does this character complete in this snippet?",
    "What, if any, goals does this character gain in this snippet that they wish to accomplish in the future?",
    "What, if any, information has this character learned in this snippet?",
    "What, if any, physical descriptions of this character are in this snippet?",
    "What, if anything, have we learned about how this character speaks from this snippet?",
]


def get_questions_to_statements(
    question_to_good_rows, filtering=False
) -> Dict[str, List[List[str]]]:
    """Iterates over datapoints from verification module and extracts the statements for each question (only if rating is >= 5 if filtering).

    :param Dict[str, List[List[Tuple[str, str, str, int]]]] question_to_good_rows:
    :param bool filtering: Whether to filter out statements with rating < 5, defaults to False.
    :return Dict[str, List[List[str]]]: dictionary of questions to list of lists of (potentially filtered) statements (one list per snippet).
    """
    question_to_all_statements = {}
    for k, v in question_to_good_rows.items():
        actual_statements = []
        for batch in v:
            # batch is a specific snippet
            per_snippet_statements = []
            for statement, snippet, character, rating in batch:
                if filtering:
                    if rating < 5:
                        continue
                per_snippet_statements.append(statement)

            actual_statements.append(per_snippet_statements)
        question_to_all_statements[k] = actual_statements
    return question_to_all_statements


def get_vectorization_for_deduplication(
    all_statements_for_fit: List[str],
) -> TfidfVectorizer:
    """Creates a TfidfVectorizer object and fits it to the given statements. Useful for deduplication of statements.

    :param List[str] all_statements_for_fit: List of statements to fit the vectorizer to.
    :return TfidfVectorizer: The fitted TfidfVectorizer object.
    """
    # Create TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # fit vectorizer for statements (probably all statements in character sheet)
    vectorizer.fit(all_statements_for_fit)

    return vectorizer


def deduplication_of_statements(
    vectorizer: TfidfVectorizer,
    all_statements_to_deduplicate: List[str],
    threshold: float = 0.9,
) -> List[List[int]]:
    """Deduplicates list of statements using cosine similarity.

    :param TfidfVectorizer vectorizer: The fitted TfidfVectorizer object.
    :param List[str] all_statements_to_deduplicate: List of statements to deduplicate.
    :param float threshold: The cosine similarity threshold for deduplication, defaults to 0.9.
    :return List[List[int]]: List of lists of indices of statements to include.
    """
    # note: we separate fit and transform so we can use the entire character sheet for fit and just deduplicate on
    # subsections of the character sheet

    # generate matrix of vectors
    tfidf_matrix = vectorizer.transform(all_statements_to_deduplicate)

    # get similarities between each sentence
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # iterate over statements and see if we had already seen it
    statement_and_whether_to_include = []
    for i, statement in enumerate(all_statements_to_deduplicate):
        is_duplicate = False
        # for every statement, see if it is a duplicate with something already seen
        for j in range(i):
            # for every prior statement
            similarity = cosine_sim[i, j]
            if similarity >= threshold:
                is_duplicate = True
                break
        should_include = not is_duplicate
        statement_and_whether_to_include.append([i, statement, should_include])
    return statement_and_whether_to_include


def deduplicate_statements_per_snippet(
    vectorizer: TfidfVectorizer, statements_per_snippet: List[List[str]]
) -> List[List[str]]:
    """Iterates over statements per snippet and deduplicates them using cosine similarity.

    :param TfidfVectorizer vectorizer: The fitted TfidfVectorizer object.
    :param List[List[str]] statements_per_snippet: List of lists of statements for each snippet.
    :return List[List[str]]: List of lists of deduplicated statements for each snippet.
    """
    flattened = flatten_list(statements_per_snippet)
    if len(flattened) == 0:
        return statements_per_snippet
    #     statement_and_whether_to_include = deduplication_of_statements(vectorizer, flattened)
    statement_and_whether_to_include = deduplication_of_statements(
        vectorizer, flattened
    )

    cur_statement_index = 0
    new_statements_per_snippet = []
    for snippet_index, statements in enumerate(statements_per_snippet):
        new_statements = []
        for statement in statements:
            should_include = statement_and_whether_to_include[cur_statement_index][-1]
            if should_include:
                new_statements.append(statement)
            cur_statement_index += 1
        new_statements_per_snippet.append(new_statements)
    return new_statements_per_snippet


def to_STATEMENT_SNIPPET_CHAR_TO_RATING(
    all_preds: pd.DataFrame,
) -> Dict[Tuple[str, str, str], int]:
    """Converts predictions from validation module to a dictionary of statement, snippet, character to rating.

    :param pd.DataFrame all_preds: DataFrame containing predictions.
    :return Dict[Tuple[str, str, str], int]: Dictionary of statement, snippet, character to rating.
    """
    STATEMENT_SNIPPET_CHAR_TO_RATING = {}
    for i, row in all_preds.iterrows():
        text = row["text"]
        pred = row["pred"]
        
        stat, snip, char = display_text_to_statement_snippet_char(text)
        if len(snip.strip()) == 0:
            # ignore these - should have been filtered out originally
            continue
        STATEMENT_SNIPPET_CHAR_TO_RATING[(stat.strip(), snip.strip(), char.strip())] = (
            pred
        )

    return STATEMENT_SNIPPET_CHAR_TO_RATING


def get_structured_data(
    STATEMENT_SNIPPET_CHAR_TO_RATING: Dict[Tuple[str, str, str], int],
    stories_input: List[List[Dict[str, str]]],
    unique_questions: List[str],
    storytype_func_split_storyid: Optional[Callable[[str], str]] = None,
    story_types: List[str] = ["default"],
    seen_stories: Optional[Set[str]] = None,
):
    """Iterates over stories and extracts statements for each question.

    :param Dict[Tuple[str, str, str], int] STATEMENT_SNIPPET_CHAR_TO_RATING: Dictionary of statement, snippet, character to rating.
    :param List[List[Dict[str, str]]] stories_input: List of stories, where each story is a list of dictionaries with 'story_id' and 'character'.
    :param List[str] unique_questions: List of unique questions.
    :param Optional[Callable[[str], str]] storytype_func_split_storyid: Function to split story_id into story type, defaults to None. Useful when calculating densities and we want to just pass in the story_id and get the story type.
    :param List[str] story_types: List of story types, defaults to ["default"].
    :param Optional[Set[str]] seen_stories: Set of stories to include from stories_input, defaults to None (include all).
    """

    storytype_to_total_num_sentences = {stype: [] for stype in story_types}
    storytype_to_total_num_sentences_filtered = {stype: [] for stype in story_types}
    storytype_to_total_num_sentences_filtered_and_dedup = {
        stype: [] for stype in story_types
    }
    storytype_to_story_names_to_filtered_and_dedup_statements = {
        stype: {} for stype in story_types
    }
    storytype_to_story_names_to_filtered_statements = {
        stype: {} for stype in story_types
    }
    storytype_to_story_names_to_unfiltered_statements = {
        stype: {} for stype in story_types
    }
    storytype_to_data_by_questions = {stype: {} for stype in story_types}

    for cur_story in tqdm(stories_input):
        # if seen_stories is not None and cur_story[0]["story_id"] not in seen_stories:
        #     all_ids_in_this_story = set([x["story_id"] for x in cur_story])
        #     print(f"skipping {cur_story[0]['story_id']} with ids {all_ids_in_this_story}")
        #     continue
        if storytype_func_split_storyid is not None:
            story_type = storytype_func_split_storyid(cur_story[0]["story_id"])
        else:
            story_type = "default"

        question_to_good_rows = {q: [] for q in unique_questions}
        for question in unique_questions:
            for snippet_datapoint in cur_story:
                snippet = snippet_datapoint["snippet"].strip()
                snippet = regexspace.sub(" ", snippet)
                # snippet = regexlines.sub(" ", snippet)
                if len(snippet) == 0:
                    # ignore these - should have been filtered out originally
                    continue
                if snippet_datapoint["question"] == question:
                    # snippet = snip["snippet"].strip()
                    char = snippet_datapoint["character"].strip()
                    sentences = snippet_datapoint["response"]
                    if type(sentences) is str:
                        sentences = get_sentences(sentences)
                    rows = []
                    for s in sentences:
                        # HARD CODED FIX FOR NOW
                        if s.strip() == "newspaper.":
                            continue
                        # For some settings (generating datasets for masking for example), the original
                        # output dataset contains statements+snippets that we ended up ignoring
                        if (
                            s.strip(),
                            snippet,
                            char,
                        ) in STATEMENT_SNIPPET_CHAR_TO_RATING:
                            pred = STATEMENT_SNIPPET_CHAR_TO_RATING[
                                (s.strip(), snippet, char)
                            ]
                            row = [s, snippet, char, pred]
                            rows.append(row)
                        else:
                            print(f"Statement {s} not found in STATEMENT_SNIPPET_CHAR_TO_RATING")

                    question_to_good_rows[question].append(rows)
        unfilt_question_to_all_statements = get_questions_to_statements(
            question_to_good_rows, filtering=False
        )
        filt_question_to_all_statements = get_questions_to_statements(
            question_to_good_rows, filtering=True
        )
        # structure should be a {question: [statements_per_snippet for snippet in snippets] }

        # all sentences that were filtered (as correct) by our classifier
        all_filtered_statements = []
        for q, statements_per_snippet in filt_question_to_all_statements.items():
            flattened = flatten_list(statements_per_snippet)
            all_filtered_statements.extend(flattened)

        if len(all_filtered_statements) != 0:
            vectorizer = get_vectorization_for_deduplication(all_filtered_statements)

            # deduplicating our character sheets
            filtered_and_deduplicated_questions_to_all_statements = {}
            for q, statements_per_snippet in filt_question_to_all_statements.items():
                new_statements_per_snippet = deduplicate_statements_per_snippet(
                    vectorizer, statements_per_snippet
                )
                filtered_and_deduplicated_questions_to_all_statements[q] = (
                    new_statements_per_snippet
                )
        else:
            # they should both be empty
            filtered_and_deduplicated_questions_to_all_statements = (
                filt_question_to_all_statements
            )

        unfilt_flattened = []
        filt_flattened = []
        filt_and_dedup_flattened = []
        # iterate over questions and accumulate flattened sentences
        for question in unique_questions:
            unfilt_lines = unfilt_question_to_all_statements[question]
            num_unfilt_lines = 0
            for b in unfilt_lines:
                for l in b:
                    unfilt_flattened.append(l)
                    num_unfilt_lines += 1

            filt_lines = filt_question_to_all_statements[question]
            num_filt_lines = 0
            for b in filt_lines:
                for l in b:
                    filt_flattened.append(l)
                    num_filt_lines += 1

            filt_and_dedup_lines = (
                filtered_and_deduplicated_questions_to_all_statements[question]
            )
            num_filt_and_dedup_lines = 0
            for b in filt_and_dedup_lines:
                for l in b:
                    filt_and_dedup_flattened.append(l)
                    num_filt_and_dedup_lines += 1

            question_type = question_to_type[question]
            if question_type in storytype_to_data_by_questions[story_type]:
                storytype_to_data_by_questions[story_type][question_type][
                    "num_gen"
                ] += num_unfilt_lines
                storytype_to_data_by_questions[story_type][question_type][
                    "num_cor"
                ] += num_filt_lines
                storytype_to_data_by_questions[story_type][question_type][
                    "num_cor_dedup"
                ] += num_filt_and_dedup_lines
            else:
                storytype_to_data_by_questions[story_type][question_type] = {
                    "num_gen": num_unfilt_lines,
                    "num_cor": num_filt_lines,
                    "num_cor_dedup": num_filt_and_dedup_lines,
                }

        storytype_to_total_num_sentences[story_type].append(len(unfilt_flattened))
        storytype_to_total_num_sentences_filtered[story_type].append(
            len(filt_flattened)
        )
        storytype_to_total_num_sentences_filtered_and_dedup[story_type].append(
            len(filt_and_dedup_flattened)
        )

        story_name = cur_story[0]["story_id"] + "-" + cur_story[0]["character"]

        storytype_to_story_names_to_filtered_and_dedup_statements[story_type][
            story_name
        ] = filtered_and_deduplicated_questions_to_all_statements
        storytype_to_story_names_to_filtered_statements[story_type][
            story_name
        ] = filt_question_to_all_statements
        storytype_to_story_names_to_unfiltered_statements[story_type][
            story_name
        ] = unfilt_question_to_all_statements

    # return the dataset at each stage
    return (
        storytype_to_total_num_sentences,
        storytype_to_total_num_sentences_filtered,
        storytype_to_total_num_sentences_filtered_and_dedup,
        storytype_to_story_names_to_filtered_and_dedup_statements,
        storytype_to_story_names_to_filtered_statements,
        storytype_to_story_names_to_unfiltered_statements,
        storytype_to_data_by_questions,
    )


def to_SNIPPET_TO_ROLE(input_original_data):
    snippet_to_role = {}
    for story in input_original_data:
        for role, og_snippet, split_snippet in story["snippets"]:
            snippet_to_role[split_snippet.strip()] = role
    return snippet_to_role


def format_character_sheet_for_writing_to_files(questions_to_statements_by_snippet):
    text_by_question = []

    for question, statements_by_snippet in questions_to_statements_by_snippet.items():
        question_text = "Question: " + question.strip()
        question_text += "\n\n"
        per_snippet_text = []
        for snippet in statements_by_snippet:
            per_snippet_text.append(" ".join(snippet))
        for snippet_index, snippet_text in enumerate(per_snippet_text):
            question_text += f"<snippet {snippet_index}>\n"
            question_text += snippet_text + "\n"
        text_by_question.append(question_text)

    output_text = ("-" * 100 + "\n").join(text_by_question)
    return output_text


def write_data(
    folder,
    storytype_to_story_names_to_filtered_and_dedup_statements,
    storytype_to_story_names_to_filtered_statements,
    storytype_to_story_names_to_unfiltered_statements,
):

    for (
        storytype,
        story_names_to_filtered_and_dedup_statements,
    ) in storytype_to_story_names_to_filtered_and_dedup_statements.items():
        for (
            name,
            questions_to_statements_by_snippet,
        ) in story_names_to_filtered_and_dedup_statements.items():
            fname_prefix = name
            if storytype != "default":
                fname_prefix = storytype + name
            filt_dedup_save_name = f"./{folder}/{fname_prefix}_postfilteringdedup.txt"
            filt_save_name = f"./{folder}/{fname_prefix}_postfiltering.txt"
            unfilt_save_name = f"./{folder}/{fname_prefix}_unfiltered.txt"

            filtered_dedup_text = format_character_sheet_for_writing_to_files(
                questions_to_statements_by_snippet
            )
            with open(filt_dedup_save_name, "w") as f:
                f.write(filtered_dedup_text)

            filtered_questions_to_statements_by_snippet = (
                storytype_to_story_names_to_filtered_statements[storytype][name]
            )
            filtered_text = format_character_sheet_for_writing_to_files(
                filtered_questions_to_statements_by_snippet
            )
            with open(filt_save_name, "w") as f:
                f.write(filtered_text)

            unfiltered_questions_to_statements_by_snippet = (
                storytype_to_story_names_to_unfiltered_statements[storytype][name]
            )
            unfiltered_text = format_character_sheet_for_writing_to_files(
                unfiltered_questions_to_statements_by_snippet
            )
            with open(unfilt_save_name, "w") as f:
                f.write(unfiltered_text)


def print_character_sheet(
    storytype_to_story_names_to_filtered_and_dedup_statements,
    storytype_to_story_names_to_filtered_statements,
    storytype_to_story_names_to_unfiltered_statements,
    output_file=None,
):
    # should only be used with one character sheet
    for (
        storytype,
        story_names_to_filtered_and_dedup_statements,
    ) in storytype_to_story_names_to_filtered_and_dedup_statements.items():

        for (
            name,
            questions_to_statements_by_snippet,
        ) in story_names_to_filtered_and_dedup_statements.items():

            unfiltered_questions_to_statements_by_snippet = (
                storytype_to_story_names_to_unfiltered_statements[storytype][name]
            )
            unfiltered_text = format_character_sheet_for_writing_to_files(
                unfiltered_questions_to_statements_by_snippet
            )
            print("Unfiltered")
            print(unfiltered_text)

            print("-" * 100)
            print("-" * 100)
            
            print("Filtered")
            filtered_questions_to_statements_by_snippet = (
                storytype_to_story_names_to_filtered_statements[storytype][name]
            )
            filtered_text = format_character_sheet_for_writing_to_files(
                filtered_questions_to_statements_by_snippet
            )
            print(filtered_text)

            print("-" * 100)
            print("-" * 100)

            filtered_dedup_text = format_character_sheet_for_writing_to_files(
                questions_to_statements_by_snippet
            )

            print("Filtered and Deduplicated")
            print(filtered_dedup_text)

            if output_file is not None:
                with open(output_file, "w") as f:
                    f.write(filtered_dedup_text)


def get_stats(
    raw_input_stories,
    prefix,
    storytype_to_total_num_sentences,
    storytype_to_total_num_sentences_filtered,
    storytype_to_total_num_sentences_filtered_and_dedup,
    storytype_to_story_names_to_filtered_and_dedup_statements,
    storytype_to_story_names_to_filtered_statements,
    storytype_to_story_names_to_unfiltered_statements,
    storytype_to_data_by_questions,
    storytype_func_split_storyid=None,
    filter_to_role=False,
    seen_stories=None,
):

    stats_by_storytype = []
    storytypes = list(storytype_to_total_num_sentences.keys())
    for storytype in storytypes:
        total_num_sentences = storytype_to_total_num_sentences[storytype]
        total_num_sentences_filtered = storytype_to_total_num_sentences_filtered[
            storytype
        ]
        total_num_sentences_filtered_and_dedup = (
            storytype_to_total_num_sentences_filtered_and_dedup[storytype]
        )
        story_names_to_filtered_statements = (
            storytype_to_story_names_to_filtered_statements[storytype]
        )
        story_names_to_filtered_and_dedup_statements = (
            storytype_to_story_names_to_filtered_and_dedup_statements[storytype]
        )
        story_names_to_unfiltered_statements = (
            storytype_to_story_names_to_unfiltered_statements[storytype]
        )

        print(storytype)

        num_stories = len(storytype_to_story_names_to_filtered_statements[storytype])
        print(storytype_to_story_names_to_filtered_statements[storytype].keys())
        print(f"num_stories", num_stories)

        # average number sentences generated for a character sheet
        avg_gen = sum([t for t in total_num_sentences]) / num_stories
        print("avg # gen per sheet", avg_gen)

        # average number filtered (correct) sentenes generated for a character sheet
        avg_cor = sum([t for t in total_num_sentences_filtered]) / num_stories
        print("avg # filt per sheet", avg_cor)

        # average number filtered (correct) sentenes generated for a character sheet
        avg_cor_dedup = (
            sum([t for t in total_num_sentences_filtered_and_dedup]) / num_stories
        )
        print("avg # filt and dedup per sheet", avg_cor_dedup)

        all_snippets_total = []
        for story in raw_input_stories:
            # if we have the splitting function then we should filter for the correct storytype
            if storytype_func_split_storyid is not None:
                cur_story_type = storytype_func_split_storyid(story["story_id"])
                if cur_story_type != storytype:
                    continue

            if seen_stories is not None and story["story_id"] not in seen_stories:
                continue

            for snippet_tuple in story["snippets"]:
                # some tuples are 3 items (includes role), some are 2 (og, subsnippet of good length)
                # the last item is always the subsnippet we fed to the model
                s = snippet_tuple[-1]
                if filter_to_role:
                    # note we only filter on datasets where there is a role in the first tuple index
                    if story["character"].strip() == snippet_tuple[0].strip():
                        all_snippets_total.append(s)
                else:
                    all_snippets_total.append(s)

        # true # sentences in input
        t_num_sentences_in_input = 0
        for snip in tqdm(all_snippets_total):
            num_sent = len(get_sentences(snip))
            t_num_sentences_in_input += num_sent
        print("t_num", t_num_sentences_in_input)

        num_generated_sentences = sum([t for t in total_num_sentences])
        # potential density - how many generated sentences in a character sheet did we produce per sentence in input
        pot_dens = num_generated_sentences / t_num_sentences_in_input
        print("pot dens", pot_dens)

        num_correct_sentences = sum([t for t in total_num_sentences_filtered])
        # density - how many correct sentences in a character sheet did we produce per sentence in input
        dens = num_correct_sentences / t_num_sentences_in_input
        print("filtered density", dens)

        num_correct_and_dedup_sentences = sum(
            [t for t in total_num_sentences_filtered_and_dedup]
        )
        # dedup density - how many correct sentences in a character sheet did we produce per sentence in input
        dens_dedup = num_correct_and_dedup_sentences / t_num_sentences_in_input
        print("filtered and dedup density", dens_dedup)

        # filtered accuracy
        acc = num_correct_sentences / num_generated_sentences
        print("filtered accuracy", acc)

        # accuracy
        acc_dedup = num_correct_and_dedup_sentences / num_generated_sentences
        print("filtered accuracy dedup", acc_dedup)

        print("-" * 100)

        rowname_for_dataset = ""
        if storytype == "default":
            rowname_for_dataset = prefix
        else:
            rowname_for_dataset = prefix + "-" + storytype

        row = {
            "storytype": rowname_for_dataset,
            "avg_gen": avg_gen,
            "avg_cor": avg_cor,
            "avg_cor_dedup": avg_cor_dedup,
            "# sentences": t_num_sentences_in_input,
            "pot density": pot_dens,
            "density_filt": dens,
            "density_filt_dedup": dens_dedup,
            "acc_filt": acc,
            "acc_filt_dedup": acc_dedup,
        }
        stats_by_storytype.append(row)

        for qtype, count_data in storytype_to_data_by_questions[storytype].items():
            num_gen_for_qtype = count_data["num_gen"]
            num_cor_for_qtype = count_data["num_cor"]
            num_cor_dedup_for_qtype = count_data["num_cor_dedup"]

            row = {
                "storytype": rowname_for_dataset + ":sub:" + qtype,
                "avg_gen": num_gen_for_qtype / num_stories,
                "avg_cor": num_cor_for_qtype / num_stories,
                "avg_cor_dedup": num_cor_dedup_for_qtype / num_stories,
                "# sentences": t_num_sentences_in_input,
                "pot density": num_gen_for_qtype / t_num_sentences_in_input,
                "density_filt": num_cor_for_qtype / t_num_sentences_in_input,
                "acc_filt": num_cor_for_qtype / num_gen_for_qtype,
                "density_filt_dedup": num_cor_dedup_for_qtype
                / t_num_sentences_in_input,
                "acc_filt_dedup": num_cor_dedup_for_qtype / num_gen_for_qtype,
            }
            stats_by_storytype.append(row)

    return stats_by_storytype


def make_dataset_write_and_get_stats(
    output_predictions_fname,
    input_original_data_fname,
    all_preds_fname,
    DATASET_NAME,
    FOLDER_NAME,
    reform_for_presnippet=False,
    FILTER_TO_ROLE=False,
    get_statistics=True,
):

    with jsonlines.open(output_predictions_fname, "r") as reader:
        output_predictions = list(reader)

    with jsonlines.open(input_original_data_fname, "r") as reader:
        input_original_data = list(reader)
    # input_original_data = input_original_data[0]

    all_preds = pd.read_csv(all_preds_fname)

    print("Loaded Data")
    print(
        len(output_predictions),
        len(input_original_data),
        len(all_preds),
        sum([len(o) for o in output_predictions]),
    )

    if reform_for_presnippet:
        reformatted_output_predictions = []
        for o in output_predictions:
            for i, line in enumerate(o):
                new_sid = line["story_id"] + "_usablesnippet_index" + str(i)
                new_datapoint = line.copy()
                new_datapoint["story_id"] = new_sid
                reformatted_output_predictions.append([new_datapoint])
        output_predictions = reformatted_output_predictions
        print("reformated")

    try:
        shutil.rmtree(FOLDER_NAME)
    except:
        pass
    os.mkdir(FOLDER_NAME)

    print("Deleted and remade folder")

    STATEMENT_SNIPPET_CHAR_TO_RATING = to_STATEMENT_SNIPPET_CHAR_TO_RATING(all_preds)

    if FILTER_TO_ROLE:
        print("Filtering to role")
        snippet_to_role = to_SNIPPET_TO_ROLE(input_original_data)
        new_output_predictions = []
        for story in output_predictions:
            new_story_output = []
            for o in story:
                #             if snippet_to_role[o['snippet'].strip()] != o['character'].strip():
                if snippet_to_role[o["snippet"].strip()] != o["character"].strip():
                    continue
                new_story_output.append(o)
            if len(new_story_output) > 0:
                new_output_predictions.append(new_story_output)
        output_predictions = new_output_predictions

        # NOT DOING THE DELETING ANYMORE, BUT WE SHOULD NEVER BE PULLING FROM NON-ROLE DATA ANYWAY
        to_delete = []
        for stat, snip, char in STATEMENT_SNIPPET_CHAR_TO_RATING.keys():
            if snippet_to_role[snip] != char:
                to_delete.append((stat, snip, char))
        for item in to_delete:
            del STATEMENT_SNIPPET_CHAR_TO_RATING[item]

    all_questions = set([s["question"] for story in output_predictions for s in story])
    seen_stories = set([s["story_id"] for story in output_predictions for s in story])
    print("# seen_stories", len(seen_stories), seen_stories)
    num_seen_in_input = 0
    for story in input_original_data:
        if story["story_id"] in seen_stories:
            num_seen_in_input += 1
    print("# seen_in_input", num_seen_in_input)
    (
        storytype_to_total_num_sentences,
        storytype_to_total_num_sentences_filtered,
        storytype_to_total_num_sentences_filtered_and_dedup,
        storytype_to_story_names_to_filtered_and_dedup_statements,
        storytype_to_story_names_to_filtered_statements,
        storytype_to_story_names_to_unfiltered_statements,
        storytype_to_data_by_questions,
    ) = get_structured_data(
        STATEMENT_SNIPPET_CHAR_TO_RATING, output_predictions, all_questions, seen_stories=seen_stories
    )

    print("Got data - writing to folder")
    write_data(
        folder=FOLDER_NAME,
        storytype_to_story_names_to_filtered_and_dedup_statements=storytype_to_story_names_to_filtered_and_dedup_statements,
        storytype_to_story_names_to_filtered_statements=storytype_to_story_names_to_filtered_statements,
        storytype_to_story_names_to_unfiltered_statements=storytype_to_story_names_to_unfiltered_statements,
    )

    if get_statistics:
        print("Getting stats")
        stats_by_story_type = get_stats(
            input_original_data,
            DATASET_NAME,
            storytype_to_total_num_sentences,
            storytype_to_total_num_sentences_filtered,
            storytype_to_total_num_sentences_filtered_and_dedup,
            storytype_to_story_names_to_filtered_and_dedup_statements,
            storytype_to_story_names_to_filtered_statements,
            storytype_to_story_names_to_unfiltered_statements,
            storytype_to_data_by_questions,
            seen_stories=seen_stories,
            filter_to_role=FILTER_TO_ROLE,
        )
        return stats_by_story_type


def get_character_sheet_for_one_snippet_data(
    output_predictions_fname,
    all_preds_fname,
    output_file=None,
):

    with jsonlines.open(output_predictions_fname, "r") as reader:
        output_predictions = list(reader)

    all_preds = pd.read_csv(all_preds_fname)

    print("Loaded Data")

    STATEMENT_SNIPPET_CHAR_TO_RATING = to_STATEMENT_SNIPPET_CHAR_TO_RATING(all_preds)

    # get all questions
    all_questions = set([s["question"] for story in output_predictions for s in story])
    
    (
        storytype_to_total_num_sentences,
        storytype_to_total_num_sentences_filtered,
        storytype_to_total_num_sentences_filtered_and_dedup,
        storytype_to_story_names_to_filtered_and_dedup_statements,
        storytype_to_story_names_to_filtered_statements,
        storytype_to_story_names_to_unfiltered_statements,
        storytype_to_data_by_questions,
    ) = get_structured_data(
        STATEMENT_SNIPPET_CHAR_TO_RATING, output_predictions, all_questions
    )

    print("Got character sheet(s) - printing")
    print_character_sheet(
        storytype_to_story_names_to_filtered_and_dedup_statements=storytype_to_story_names_to_filtered_and_dedup_statements,
        storytype_to_story_names_to_filtered_statements=storytype_to_story_names_to_filtered_statements,
        storytype_to_story_names_to_unfiltered_statements=storytype_to_story_names_to_unfiltered_statements,
        output_file=output_file,
    )
