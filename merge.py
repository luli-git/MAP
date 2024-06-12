from task_vectors import TaskVector
import os
from copy import deepcopy


def get_merged_model(
    pretrained_checkpoint, finetuned_checkpoints, scaling, save_merged_model_dir=None
):
    """
    Merge task vectors from multiple models with different scaling factors.
    :param pretrained_checkpoint: Path to the pretrained model checkpoint.
    :param finetuned_checkpoints: List of paths to the finetuned model checkpoints.
    :param scaling: List of scaling factors for each model.
    """
    task_vectors = [
        TaskVector(pretrained_checkpoint, ft).scale(scale)
        for ft, scale in zip(finetuned_checkpoints, scaling)
    ]
    task_vector_sum = sum(task_vectors)
    image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=1)
    if save_merged_model_dir:
        if not os.path.exists(save_merged_model_dir):
            os.makedirs(save_merged_model_dir, exist_ok=True)
        ft_path = os.path.join(save_merged_model_dir, "finetuned.pt")
        image_encoder.save(ft_path)
    return image_encoder
