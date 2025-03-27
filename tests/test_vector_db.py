import sys,os
sys.path.append(os.getcwd())
from ReasonFlux.template_matcher.service import OpenAIEmbeddingService

def test_openai_embedding():
    embedding_service = OpenAIEmbeddingService(
        api_key="sk-xx",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="text-embedding-v3"
    )
    encoded = embedding_service.encode("Hello, world!")
    print(encoded)

if __name__ == "__main__":
    test_openai_embedding()