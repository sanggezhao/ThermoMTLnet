"""
function
"""
import torch
from torchmetrics.functional.regression import pearson_corrcoef
from fastprop.model import MultiTaskRegressor, MultiTask_PINNloss
from fastprop.data import fastpropDataLoader, fastpropDataset, standard_scale
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import KFold
import numpy as np
from fastprop.model import train_and_test
from astartes.molecules import train_val_test_split_molecules

def calculate_mean_scores(results_list, tasks=4):
    means = []
    for task_idx in range(tasks):
        task_key = f"test_r_score_task{task_idx}"
        task_scores = [result[task_key] for result in results_list]
        mean = sum(task_scores) / len(task_scores)
        means.append(mean)
    return means


def create_loaders_and_models(descriptors, targets, train_indexes, val_indexes, test_indexes):

    descriptors[train_indexes], feature_means, feature_vars = standard_scale(descriptors[train_indexes])
    descriptors[val_indexes] = standard_scale(descriptors[val_indexes], feature_means, feature_vars)
    descriptors[test_indexes] = standard_scale(descriptors[test_indexes], feature_means, feature_vars)
    targets[train_indexes], targets_means, targets_vars = standard_scale(targets[train_indexes])
    targets[val_indexes] = standard_scale(targets[val_indexes], targets_means, targets_vars)
    targets[test_indexes] = standard_scale(targets[test_indexes], targets_means, targets_vars)

    train_dataloader = fastpropDataLoader(fastpropDataset(descriptors[train_indexes], targets[train_indexes]), shuffle=True, batch_size=32)
    val_dataloader = fastpropDataLoader(fastpropDataset(descriptors[val_indexes], targets[val_indexes]), batch_size=128)
    test_dataloader = fastpropDataLoader(fastpropDataset(descriptors[test_indexes], targets[test_indexes]), batch_size=1024)

    baseline_model = MultiTask_PINNloss(
        fnn_layers=0,
        hidden_size=350,
        feature_means=feature_means,
        feature_vars=feature_vars,
        target_means=targets_means,
        target_vars=targets_vars,
        learning_rate=0.0001,
    )
    real_model = MultiTask_PINNloss(
        clamp_input=True,
        fnn_layers=3,
        hidden_size=1800,
        feature_means=feature_means,
        feature_vars=feature_vars,
        target_means=targets_means,
        target_vars=targets_vars,
        learning_rate=0.0001,
    )
    return baseline_model, real_model, train_dataloader, val_dataloader, test_dataloader

# 交叉验证
def cross_validate_fastprop(
    smiles_arr: np.ndarray,
    descriptors_arr: np.ndarray,
    targets_arr: np.ndarray,
    num_tasks,
    outdir: str,
):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    real_predictions, baseline_predictions, fold_truth = [], [], []
    for train_indexes, test_indexes in kf.split(np.arange(len(smiles_arr))):
        descriptors = torch.tensor(descriptors_arr, dtype=torch.float32)
        targets = torch.tensor(targets_arr, dtype=torch.float32)
        baseline_model, real_model, train_dataloader, _, test_dataloader = create_loaders_and_models(
            descriptors, targets, train_indexes, np.array([]), test_indexes
        )
        tensorboard_logger = TensorBoardLogger(
            outdir,
            name="tensorboard_logs",
            default_hp_metric=False,
        )
        trainer = Trainer(
            max_epochs=30,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=tensorboard_logger,
            log_every_n_steps=1,
            enable_checkpointing=False,
            check_val_every_n_epoch=1,
        )
        trainer.fit(real_model, train_dataloader)
        # the predict method will auto-magically scale the descriptors and then undo the scaling of the outputs
        # in-place, so we provide a fresh copy for each call
        descriptors = torch.tensor(descriptors_arr, dtype=torch.float32)
        targets = torch.tensor(targets_arr, dtype=torch.float32)
        test_dataloader = fastpropDataLoader(fastpropDataset(descriptors[test_indexes],
                                                             targets[test_indexes]), batch_size=1024)
        # test_predictions: np.ndarray = trainer.predict(real_model, test_dataloader)[0].numpy().ravel()
        test_predictions: np.ndarray = torch.cat(trainer.predict(real_model, test_dataloader), dim=0).numpy()
        real_predictions.extend(test_predictions)
        tensorboard_logger = TensorBoardLogger(
            outdir,
            name="tensorboard_logs",
            default_hp_metric=False,
        )
        trainer = Trainer(
            max_epochs=30,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=tensorboard_logger,
            log_every_n_steps=1,
            enable_checkpointing=False,
            check_val_every_n_epoch=1,
        )
        trainer.fit(baseline_model, train_dataloader)
        descriptors = torch.tensor(descriptors_arr, dtype=torch.float32)
        targets = torch.tensor(targets_arr, dtype=torch.float32)
        test_dataloader = fastpropDataLoader(fastpropDataset(descriptors[test_indexes],
                                                             targets[test_indexes]), batch_size=8)
        # test_predictions: np.ndarray = trainer.predict(baseline_model, test_dataloader)[0].numpy().ravel()
        test_predictions: np.ndarray = torch.cat(trainer.predict(baseline_model, test_dataloader), dim=0).numpy()
        baseline_predictions.extend(test_predictions)
        # fold_truth.extend(targets[test_indexes].detach().clone().numpy().ravel())
        fold_truth.extend(targets[test_indexes].detach().clone().numpy())

    real = torch.tensor(real_predictions)  # shape: (n, 6)
    baseline = torch.tensor(baseline_predictions)
    truth = torch.tensor(fold_truth)
    return (
        # pearson_corrcoef(torch.tensor(real_predictions), torch.tensor(fold_truth)),
        # pearson_corrcoef(torch.tensor(baseline_predictions), torch.tensor(fold_truth)),
        [pearson_corrcoef(real[:, i], truth[:, i]).item() for i in range(real.shape[1])],
        [pearson_corrcoef(baseline[:,i], truth[:, i]).item() for i in range(baseline.shape[1])],
    )


def replicate_fastprop(
    smiles_arr: np.ndarray,
    descriptors_arr: np.ndarray,
    targets_arr: np.ndarray,
    num_tasks: int,
    outdir: str,
):
    baseline_results, fastprop_results = [], []
    for i in range(5):
        # get a fresh copy of the input data for re-scaling
        descriptors = torch.tensor(descriptors_arr, dtype=torch.float32)
        targets = torch.tensor(targets_arr, dtype=torch.float32)

        # split the data using kmeans clustering on the molecular fingerprint
        *_, train_indexes, val_indexes, test_indexes = train_val_test_split_molecules(
            smiles_arr,
            train_size=0.7,
            val_size=0.1,
            test_size=0.2,
            sampler="kmeans",
            random_state=42 + i,   # 42
            return_indices=True,
        )

        val_targets = targets[val_indexes]
        test_targets = targets[test_indexes]

        valid_val_samples = (~torch.isnan(val_targets)).sum(dim=0)
        valid_test_samples = (~torch.isnan(test_targets)).sum(dim=0)

        if (valid_val_samples < 2).any() or (valid_test_samples < 2).any():
            print(f"Skip i={i}, insufficient samples in val/test.")
            continue  

        baseline_model, real_model, train_dataloader, val_dataloader, test_dataloader = create_loaders_and_models(
            descriptors, targets, train_indexes, val_indexes, test_indexes
        )
        test_results, validation_results = train_and_test(outdir, baseline_model, train_dataloader, val_dataloader, test_dataloader, quiet=True)
        baseline_results.append(test_results[0])
        test_results, validation_results = train_and_test(outdir, real_model, train_dataloader, val_dataloader, test_dataloader, quiet=True)
        fastprop_results.append(test_results[0])
    return (
        calculate_mean_scores(fastprop_results, num_tasks),
        calculate_mean_scores(baseline_results, num_tasks),
    )
