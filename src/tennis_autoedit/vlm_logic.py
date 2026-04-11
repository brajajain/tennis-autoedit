import mlx_vlm
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image
from typing import List, Tuple, Any

def load_vlm(model_path: str) -> Tuple[Any, Any, Any]:
    """
    Load the VLM model, processor, and config.
    """
    model, processor = mlx_vlm.load(model_path)
    config = load_config(model_path)
    return model, processor, config

def check_point_in_progress(
    model: Any,
    processor: Any,
    config: Any,
    images: List[Image.Image],
    max_tokens: int = 10
) -> bool:
    """
    Perform inference on a sequence of images to determine if a tennis point is in progress.
    """
    prompt = (
        "Is a tennis point currently in progress? "
        "Look for players in a ready stance or actively hitting the ball. "
        "Ignore players walking, sitting, or picking up balls. "
        "Answer 'Yes' or 'No'."
    )

    formatted_prompt = apply_chat_template(
        processor, config, prompt, num_images=len(images)
    )

    output = mlx_vlm.generate(
        model,
        processor,
        formatted_prompt,
        images,
        max_tokens=max_tokens,
        verbose=False
    )

    # Simple normalization of the output
    # mlx_vlm.generate returns a string directly
    answer = output.strip().lower()
    return "yes" in answer
