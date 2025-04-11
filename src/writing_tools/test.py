if __name__ == "__main__":
    from writing_tools import SimpleSuggestionGenerator, OllamaInferenceModel

    suggestion_generator = SimpleSuggestionGenerator(OllamaInferenceModel())

    text = """
    At Ericsson NL, a major component of performance monitoring is change evaluation, 
    i.e. evaluation of the networks performance after the introduction of a software 
    or hardware related change. Engineers are tasked with monitoring KPI data before 
    and after a change has been introduced, and concluding whether and where the change 
    had an impact on performance.
    """

    prediction = suggestion_generator.predict(text, -1)

    print(prediction)