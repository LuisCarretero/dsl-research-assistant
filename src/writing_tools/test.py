if __name__ == "__main__":
    
    from writing_tools import SimpleSuggestionGenerator, HFInferenceModel, OllamaInferenceModel
    import os
    from dotenv import load_dotenv

    load_dotenv()

    MODEL_DIR = os.environ.get("MODELS_DIRECTORY")

    suggestion_generator = SimpleSuggestionGenerator(HFInferenceModel(
        MODEL_DIR + "\\deepseek-r1-distill-qwen-1.5B\\model",
        MODEL_DIR + "\\deepseek-r1-distill-qwen-1.5B\\tokenizer"
    ))
    suggestion_generator = SimpleSuggestionGenerator(OllamaInferenceModel())

    text = """
    I. CHANGE POINT DETECTION LITERATURE REVIEW
    CPD is a vastly researched field of time series analysis,
    with early advances dating as far back as the 50s [1]. The
    applications of it are also far reaching. Aside from telecom-
    munications [2], [3], CPD has been applied in fields such as
    finance and marked analysis [4], [5] gene expression data [6],
    [7] satellite time series data [8] and many more.
    It is important to make the distinction between online and
    offline CPD [1]. Online CPD attempts to find changes in real
    time, while offline CPD does so after obtaining all the required
    data. In this thesis, we will be focusing on the latter, as the
    change evaluation in Ericsson is done after the fact.
    Much research has been done in exploring parametric
    methods. Some of the first successful examples have used
    the CUSUM approach [9], which looks for abrupt changes of
    the cumulative sum of residuals given an assumed model. A
    popular approach is the Generalized Likelihood Ratio (GLR)
    procedure [10], which attempts to maximize likelihood ratios
    between segments. Bayesian Model Averaging (BMA) has also
    been successfully employed for offline CPD [8].
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

