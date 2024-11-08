import numpy as np
from tqdm import tqdm
import pandas as pd
import torch

def sigmoid(x):
    # util sigmoid function
    return np.exp(-np.logaddexp(0, -x))

def get_model_predictions_across_dataset(model, tokenizer, dataset) -> pd.DataFrame:
    """Iterate over the dataset, tokenize the text, get the model predictions from logits

    :param model: the model to use
    :param tokenizer: should be tokenizer corresponding to model
    :param dataset: the dataset to use
    :return: a dataframe with the text, prediction, and softmax results
    """
    responses = []
    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        
        text = row['text']

        # due to some tokenization issues, we replace the story section with a space
        # this is a hacky fix, but necessary for the trained model (prompted models it shouldn't make a big difference)
        text.replace("Story Section:\n", "Story Section:    \n")

        input_ids = tokenizer([text], return_tensors="pt").input_ids.to(
            model.device
        )
        
        logits = model(input_ids=input_ids).logits[0, -1]
        
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("1").input_ids[-1]],
                        logits[tokenizer("2").input_ids[-1]],
                        logits[tokenizer("3").input_ids[-1]],
                        logits[tokenizer("4").input_ids[-1]],
                        logits[tokenizer("5").input_ids[-1]],
                    ]
                ).float(),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )

        # unnecessary but nice for debugging, but we use the sigmoid function to get a probability
        # we return the softmax results for later use
        sigmoid_results = sigmoid(probs)
        
        pred = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}[np.argmax(sigmoid_results)]
                
        responses.append([text, pred, list(probs)])
    return pd.DataFrame(responses, columns=["text", "pred", "softmax_results"])