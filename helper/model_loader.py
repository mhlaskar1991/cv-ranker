import logging
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForCausalLM


def load_model(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "t5" in model_name:
        logging.debug("Loading T5 model")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="cpu"
        )

    elif "mistral" in model_name:
        logging.debug("Loading Mistral model")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu"
        )

    else:
        raise Exception("No model name specified")

    return tokenizer, model