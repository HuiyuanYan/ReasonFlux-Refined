import sys,os
import json
sys.path.append(os.getcwd())
from ReasonFlux.reason_flux import ReasonFlux

def test_reason_flux():
    # a math problem
    problem = r"Given a sequence {aₙ} satisfying a₁=3, and aₙ₊₁=2aₙ+5 (n≥1), find the general term formula aₙ"
    reason_flux = ReasonFlux(
        inference_config_path="ReasonFlux/config/agent/inference.yaml",
        navigator_config_path="ReasonFlux/config/agent/navigator.yaml",
        hierarchical_database_config_path="ReasonFlux/config/database/database.yaml"
    )

    meta_data = reason_flux.run(problem)
    if meta_data:
        with open("output/meta_data.json", "w") as f:
            json.dump(meta_data, f, indent=4)

if __name__ == "__main__":
    test_reason_flux()