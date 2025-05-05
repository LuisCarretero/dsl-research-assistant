from writing_tools import LitLLMLiteratureReviewGenerator, OllamaInferenceModel, HFClientInferenceModel
from dotenv import load_dotenv
import os
import json

load_dotenv()

if __name__ == "__main__":
    CITATION_DIR = os.environ.get("CITATION_DIR")
    HF_TOKEN = os.environ.get("HF_TOKEN")

    print("WELCOME TO THE LITERATURE REVIEW GENERATION DEMO")
    abstract = input("Provide an abstract or summary of a the paper whose \"Related Work\" section you'd like to write: ")
    if abstract == "":
        abstract = """
        Visual odometry is an ill-posed problem and utilized in
        many robotics applications, especially automated driving
        for mapless navigation. Recent applications have shown
        that deep models outperform traditional approaches especially
        in localization accuracy and furthermore significantly
        reduce catastrophic failures. The disadvantage of most of
        these models is a strong dependence on high-quantity and
        high-quality ground truth data. However, accurate and
        dense depth ground truth data for real world datasets is difficult
        to obtain. As a result, deep models are often trained on
        synthetic data which introduces a domain gap. We present
        a weakly supervised approach to overcome this limitation.
        Our approach uses estimated optical flow for training that
        can be generated without the need for high-quality dense
        depth ground truth. Instead, it only requires ground truth
        poses and raw camera images for training. In the experiments,
        we show that our approach enables deep visual
        odometry to be efficiently trained on the target domain (real
        data) while achieving state-of-the-art performance on the
        KITTI dataset.
        """
        paper = json.load(open(os.path.join(CITATION_DIR, "Abouee_Weakly_Supervised_End2End_Deep_Visual_Odometry_CVPRW_2024_paper.json"), "r"))
        reference_abstracts = [paper["references"][i]["abstract"] for i in paper["references"]]
    else:
        reference_abstracts = []
        reference_abstract = ""
        while reference_abstract.lower() != "done":
            reference_abstract = input("Provide the abstract of a reference paper (if you are done, write \"done\"): ")
            reference_abstracts.append(reference_abstract)

    inference_model = HFClientInferenceModel(provider="novita", api_key=HF_TOKEN)
    inference_model.set_default_call_kwargs(model="deepseek-ai/DeepSeek-R1")
    related_work_generator = LitLLMLiteratureReviewGenerator(inference_model)

    print("Writing the Related Work...")

    related_work = related_work_generator.predict(abstract, reference_abstracts)

    print(f"RELATED WORK\n{related_work}")