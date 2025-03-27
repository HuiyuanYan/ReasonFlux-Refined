import pytest
import sys,os
sys.path.append(os.getcwd())
from ReasonFlux.template_matcher import HierarchicalVectorDatabase
import numpy as np

# 测试数据：数学分层知识点
test_data = {
    "Mathematics": {
        "Algebra": {
            "Linear Algebra": {
                "Matrices": "A rectangular array of numbers.",
                "Vectors": "Quantities with magnitude and direction."
            },
            "Abstract Algebra": {
                "Groups": "A set equipped with an operation that combines any two elements to form a third element.",
                "Rings": "A set equipped with two operations (addition and multiplication) satisfying certain properties."
            }
        },
        "Calculus": {
            "Differential Calculus": {
                "Derivatives": "The rate of change of a function.",
                "Limits": "The value that a function approaches as the input approaches some value."
            },
            "Integral Calculus": {
                "Definite Integrals": "The signed area under a curve.",
                "Indefinite Integrals": "The antiderivative of a function."
            }
        },
        "Geometry": {
            "Euclidean Geometry": {
                "Triangles": "Three-sided polygons.",
                "Circles": "A set of points equidistant from a central point."
            },
            "Analytic Geometry": {
                "Conic Sections": "Curves obtained by intersecting a cone with a plane.",
                "Coordinate Systems": "Systems for specifying points using coordinates."
            }
        }
    }
}

# 测试嵌入服务参数
embedding_params = {
    "api_key": "sk-xx",
    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1", 
    "model": "text-embedding-v3",
    "provider": "openai"
}

# 测试函数
def test_hierarchical_database():
    db = HierarchicalVectorDatabase(
        data_dir="test_data",
        persist=True,
        embedding_params=embedding_params
    )
    
    db.add_recursive_dict(test_data)
    
    assert db.max_level == 4, "Max level should be 4"
    for i in range(db.max_level):
        collection_name = f"level_{i}"
        assert collection_name in db.collections, f"Collection {collection_name} should exist"
    
    test_embedding = db.embedding_service.encode("test query")
    assert isinstance(test_embedding, np.ndarray), "Embedding should be a list"
    
    # 测试分层搜索
    queries = ["Mathematics", "Calculus", "Differential Calculus", "Derivatives"]
    top_k_per_level = [2, 2, 2, 2]
    weight_per_level = [1.0, 1.0, 1.0, 1.0]
    results = db.hierarchical_search(
        queries=queries,
        top_k_per_level=top_k_per_level,
        weight_per_level=weight_per_level,
        final_count=1
    )
    print(results)
    
    assert results is not None, "Search results should not be None"
    assert len(results) == 1, "Should return one result"
    assert results[0]["meta_data"]["data"] == "The rate of change of a function.", "Search result should match the expected value"
    
    # 清理测试数据
    db.clear()

# 运行测试
if __name__ == "__main__":
    pytest.main([__file__])