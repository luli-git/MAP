from task_vectors.src.task_vectors import TaskVector
import os
from copy import deepcopy
import torch


def tv_merge_to_pretrain(pretrained_model, vector, scaling_coef=1.0):
    """Apply a task vector to a pretrained model."""
    with torch.no_grad():
        new_state_dict = {}
        pretrained_state_dict = pretrained_model.state_dict()
        for key in pretrained_state_dict:
            if key not in vector:
                print(
                    f"Warning: key {key} is present in the pretrained state dict but not in the task vector"
                )
                continue
            new_state_dict[key] = (
                pretrained_state_dict[key] + scaling_coef * vector[key]
            )
    pretrained_model.load_state_dict(new_state_dict, strict=False)
    return pretrained_model


def get_merged_model_from_vectors(
    vectors, pretrained_model, scaling, save_merged_model_dir=None
):
    """
    Merge task vectors from multiple models with different scaling factors.
    :param pretrained_checkpoint: Path to the pretrained model checkpoint.
    :param finetuned_checkpoints: List of paths to the finetuned model checkpoints.
    :param scaling: List of scaling factors for each model.
    """
    task_vectors = [
        TaskVector(vector=vector).scale(scale)
        for vector, scale in zip(vectors, scaling)
    ]
    task_vector_sum = sum(task_vectors)
    # image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=1)
    image_encoder = tv_merge_to_pretrain(
        deepcopy(pretrained_model), task_vector_sum.vector, scaling_coef=1
    )
    if save_merged_model_dir:
        if not os.path.exists(save_merged_model_dir):
            os.makedirs(save_merged_model_dir, exist_ok=True)
        ft_path = os.path.join(save_merged_model_dir, "finetuned.pt")
        image_encoder.save(ft_path)
    return image_encoder


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
