from writing_tools import OllamaInferenceModel, LitLLMLiteratureReviewGenerator

references = ["This is a paper that proposes the transformer architecture", "This is a paper about writing literature reviews with LLMs", "This is another paper about writing literature reviews with LLMs"]
reference_ids = [5, 3, 16]
abstract = "This paper proposes a novel approach for writing literature reviews with LLMs"

inference_model = OllamaInferenceModel()
inference_model.set_default_call_kwargs(model="deepseek-r1:7B")

model = LitLLMLiteratureReviewGenerator(inference_model)

print(model.predict_entire(abstract, references, reference_ids))