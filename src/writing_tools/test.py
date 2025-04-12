if __name__ == "__main__":
    
    from writing_tools import SimpleSuggestionGenerator, HFInferenceModel, OllamaInferenceModel
    import os
    from dotenv import load_dotenv

    load_dotenv()

    MODEL_DIR = os.environ.get("MODELS_DIRECTORY")

    #suggestion_generator = SimpleSuggestionGenerator(HFInferenceModel(
    #    MODEL_DIR + "\\deepseek-r1-distill-qwen-1.5B\\model",
    #    MODEL_DIR + "\\deepseek-r1-distill-qwen-1.5B\\tokenizer"
    #))
    suggestion_generator = SimpleSuggestionGenerator(OllamaInferenceModel())

    text = """
    2 Background
    The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU
    [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building
    block, computing hidden representations in parallel for all input and output positions. In these models,
    the number of operations required to relate signals from two arbitrary input or output positions grows
    in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes
    it more difficult to learn dependencies between distant positions [12]. In the Transformer this is
    reduced to a constant number of operations, albeit at the cost of reduced effective resolution due
    to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as
    described in section 3.2.
    Self-attention, sometimes called intra-attention is an attention mechanism relating different positions
    of a single sequence in order to compute a representation of the sequence.
    """

    prediction = suggestion_generator.predict(text, -1)

    print(prediction)
    
    '''
    # Download model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from dotenv import load_dotenv
    import os

    from writing_tools import HFInferenceModel

    load_dotenv()

    MODEL_DIR = os.environ.get("MODELS_DIRECTORY")

    #model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    #model.save_pretrained(MODEL_DIR + "\\deepseek-r1-distill-qwen-1.5B\\model")
    #tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    #tokenizer.save_pretrained(MODEL_DIR + "\\deepseek-r1-distill-qwen-1.5B\\tokenizer")

    inference_model = HFInferenceModel(
        MODEL_DIR + "\\deepseek-r1-distill-qwen-1.5B\\model",
        MODEL_DIR + "\\deepseek-r1-distill-qwen-1.5B\\tokenizer"
    )

    print(inference_model.predict("Paris is the capital of"))
    '''

