from transformers import EsmForMaskedLM, EsmTokenizer

def get_plm(model_name):
    """
    :param model_name:String
    :return: pLM model
    """
    model_name = "facebook/esm2_t6_8M_UR50D"  # Or another ESM-2 variant
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)  # Or EsmModel for embeddings
    return model, tokenizer