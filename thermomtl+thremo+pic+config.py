import os
import torch
import pandas as pd
import numpy as np
from rdkit.Chem import MolFromSmiles
from pathlib import Path
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold
from torchmetrics.functional.regression import pearson_corrcoef
from torchmetrics.functional.regression import r2_score, mean_absolute_error
from astartes.molecules import train_val_test_split_molecules
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from ThermoMTLnet.model import train_and_test
from ThermoMTLnet.metrics import SCORE_LOOKUP
from ThermoMTLnet.descriptors import get_descriptors
from ThermoMTLnet.defaults import ALL_2D
from ThermoMTLnet.io import load_saved_descriptors
from ThermoMTLnet.io import read_input_csv
from ThermoMTLnet.model import MultiTaskRegressor, MultiTask_PINNloss, MultiTask_PINN_config
from ThermoMTLnet.data import fastpropDataLoader, fastpropDataset, standard_scale

target_names = {
    "Enthalpy": "Enthalpy",
    "Entropy": "Entropy",
    "Heat_comb": "Combustion_heat",
    "Heat_expl": "Explosion_heat",
}

targets, smiles = read_input_csv(
    "thermodb_300.csv",
    "SMILES",
    target_names.values(),
)

cache_file = "thermodb_descriptors300.csv"
if os.path.exists(cache_file):
    descriptors = load_saved_descriptors(cache_file)
else:
    descriptors = get_descriptors(
        cache_file,
        ALL_2D,
        list(map(MolFromSmiles, smiles)),
    ).to_numpy(dtype=np.float32)

cols_with_nan = np.isnan(descriptors).any(axis=0)
descriptors = descriptors[:, ~cols_with_nan]
print(f"descriptors_nonan.shape: {descriptors.shape}")

constant_columns = np.all(descriptors == descriptors[0, :], axis=0)
num_true_columns = np.sum(constant_columns)
descriptors = descriptors[:, ~constant_columns]
print(f"descriptors_noconstant.shape: {descriptors.shape}")

descriptors_df = pd.DataFrame(descriptors)
correlation_matrix = descriptors_df.corr(method='pearson')
# Dropping correlation with high correlation, 0.6 0.7 0.8 0.9 0.92 0.95 0.98 0.99
threshold = 0.99
high_correlation_pairs = (correlation_matrix > threshold).sum() - 1
to_drop_descriptors = [column for column in correlation_matrix.columns if high_correlation_pairs[column] > 0]
descriptors = descriptors_df.drop(columns=to_drop_descriptors)
print(f"descriptors_pcc.shape: {descriptors.shape}")

def calculate_feature_importance(descriptors, targets, target_names):
    """
    Calculate MIC and mutual information for each feature-target pair.

    Args:
        descriptors: DataFrame of molecular descriptors
        targets: Array of target values (n_samples x n_tasks)
        target_names: Dictionary mapping task names to descriptions

    Returns:
        Dictionary containing MIC and MI results for each task
    """
    results = {}

    task_names = list(target_names.keys())

    for task_num in range(targets.shape[1]):
        task_name = task_names[task_num]  
        task_mask = ~np.isnan(targets[:, task_num])
        X_task = descriptors.iloc[task_mask]
        y_task = targets[task_mask, task_num]

        # Calculate Mutual Information
        mi = mutual_info_regression(X_task, y_task, random_state=42)

        # Store results using the task name as key
        results[task_name] = {
            'Mutual_Information': mi,
            'Descriptors': X_task.columns
        }

    return results

def select_important_features(results, percentile=70):
    important_features = {}

    for task_name, task_results in results.items():
        mi_scores = task_results['Mutual_Information']
        descriptors = task_results['Descriptors']

        # 计算阈值
        mi_threshold = np.percentile(mi_scores, percentile)

        # 获取重要特征
        important_mask = mi_scores > mi_threshold
        important_descriptors = descriptors[important_mask]

        important_features[task_name] = important_descriptors.tolist()

    return important_features

feature_importance_results = calculate_feature_importance(descriptors, targets, target_names)
important_features = select_important_features(feature_importance_results)
all_important_features = set()
for features in important_features.values():
    all_important_features.update(features)
print(f"\n所有任务中的独特重要特征总数: {len(all_important_features)}")
descriptors = descriptors[list(all_important_features)].to_numpy()
print(f"descriptors_MI.shape: {descriptors.shape}")

cache_file2 = "../mutitask/handmade/theromo_newdescriptors_with_derived_features.csv"
descriptors2 = load_saved_descriptors(cache_file2)
descriptors = np.hstack([descriptors2, descriptors])
print(f"descriptors_withhandmade.shape: {descriptors.shape}")


def r_score(truth: torch.Tensor, prediction: torch.Tensor, ignored: None, multitask: bool = False):
    return pearson_corrcoef(prediction, truth)

SCORE_LOOKUP["regression"] = (r_score,) + SCORE_LOOKUP["regression"]


def calculate_mean_scores(results_list, tasks=4):
    """计算指定任务的平均分数，等效于pandas的describe().loc['mean']"""
    means = []
    for task_idx in range(tasks):
        task_key = f"test_r_score_task{task_idx}"
        task_scores = [result[task_key] for result in results_list]
        mean = sum(task_scores) / len(task_scores)
        means.append(mean)
    return means

def create_loaders_and_models(descriptors, targets, train_indexes, val_indexes, test_indexes):

    # re-scale the features and the targets
    descriptors[train_indexes], feature_means, feature_vars = standard_scale(descriptors[train_indexes])
    descriptors[val_indexes] = standard_scale(descriptors[val_indexes], feature_means, feature_vars)
    descriptors[test_indexes] = standard_scale(descriptors[test_indexes], feature_means, feature_vars)
    targets[train_indexes], targets_means, targets_vars = standard_scale(targets[train_indexes])
    targets[val_indexes] = standard_scale(targets[val_indexes], targets_means, targets_vars)
    targets[test_indexes] = standard_scale(targets[test_indexes], targets_means, targets_vars)

    # initialize dataloaders and model, then train
    train_dataloader = fastpropDataLoader(fastpropDataset(descriptors[train_indexes], targets[train_indexes]), shuffle=True, batch_size=32)
    val_dataloader = fastpropDataLoader(fastpropDataset(descriptors[val_indexes], targets[val_indexes]), batch_size=128)
    test_dataloader = fastpropDataLoader(fastpropDataset(descriptors[test_indexes], targets[test_indexes]), batch_size=1024)

    # train a linear baseline _and_ a 'real' model
    baseline_model = MultiTask_PINN_config(
        fnn_layers=0,
        hidden_size=737,
        feature_means=feature_means,
        feature_vars=feature_vars,
        target_means=targets_means,
        target_vars=targets_vars,
        learning_rate=0.0001,
        num_samples=len(train_dataloader.dataset),
    )
    real_model = MultiTask_PINN_config(
        clamp_input=True,
        fnn_layers=3,
        hidden_size=1800,
        feature_means=feature_means,
        feature_vars=feature_vars,
        target_means=targets_means,
        target_vars=targets_vars,
        learning_rate=0.001,
        num_samples=len(train_dataloader.dataset),
    )
    return baseline_model, real_model, train_dataloader, val_dataloader, test_dataloader

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

        descriptors = torch.tensor(descriptors_arr, dtype=torch.float32)
        targets = torch.tensor(targets_arr, dtype=torch.float32)
        test_dataloader = fastpropDataLoader(fastpropDataset(descriptors[test_indexes],
                                                             targets[test_indexes]), batch_size=1024)
        test_predictions: np.ndarray = torch.cat(trainer.predict(real_model, test_dataloader), dim=0).numpy()
        real_predictions.extend(test_predictions)
        fold_truth.extend(targets[test_indexes].detach().clone().numpy())

    pred = torch.tensor(real_predictions)
    truth = torch.tensor(fold_truth)

    return pred, truth

def plot_parity_plots(y_true, y_pred, target_names, save_path=None):
    y_pred = y_pred.cpu().numpy() 
    y_true = y_true.cpu().numpy()
    n_tasks = y_true.shape[1]
    task_keys = list(target_names.keys())

    plt.figure(figsize=(8, 8))

    colors = ['steelblue', 'coral', 'mediumseagreen', 'goldenrod']

    global_min = np.inf
    global_max = -np.inf

    for i in range(n_tasks):
        mask = ~(np.isnan(y_true[:, i]) | np.isnan(y_pred[:, i]))
        true_vals = y_true[mask, i]
        pred_vals = y_pred[mask, i]

        if len(true_vals) > 0:
            current_min = min(true_vals.min(), pred_vals.min())
            current_max = max(true_vals.max(), pred_vals.max())
            global_min = min(global_min, current_min)
            global_max = max(global_max, current_max)

    for i in range(n_tasks):
        mask = ~(np.isnan(y_true[:, i]) | np.isnan(y_pred[:, i]))
        true_vals = y_true[mask, i]
        pred_vals = y_pred[mask, i]

        if len(true_vals) > 0:
            plt.scatter(true_vals, pred_vals, alpha=0.6, s=20,
                        color=colors[i], label=task_keys[i])

    plt.plot([global_min, global_max], [global_min, global_max])

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.gca().set_aspect('equal', adjustable='box')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


task_mask = ~np.isnan(targets).any(axis=1)
print(f"过滤后的样本数: {np.sum(task_mask)}")
print(f"每个任务的有效样本数:")
for i, name in enumerate(target_names.keys()):
    valid_count = np.sum(~np.isnan(targets[task_mask, i]))
    print(f"  {name}: {valid_count}")

print(f"targets shape: {targets.shape}")
print(f"targets中NaN的比例: {np.isnan(targets).mean():.2%}")
descriptors = descriptors.astype(np.float32)

pred, truth = cross_validate_fastprop(
    smiles_arr=smiles[task_mask],
    descriptors_arr=descriptors[task_mask, :],
    targets_arr=targets[task_mask],
    num_tasks=len(target_names),
    outdir=Path("adme_output_multitask_interpolation"),
)

r2 = [r2_score(truth[:, i], pred[:, i]).item() for i in range(pred.shape[1])]
mae = [mean_absolute_error(pred[:, i], truth[:, i]).item() for i in range(pred.shape[1])]
nmae = [mae[i] / (truth[:, i].max() - truth[:, i].min()).item() for i in range(pred.shape[1])]
pcc = [pearson_corrcoef(truth[:, i], pred[:, i]).item() for i in range(pred.shape[1])]

metrics = {
        "r2": r2,
        "mae": mae,
        "nmae": nmae,
        "pcc": pcc,
    }

# 制表
table = tabulate(
    headers=["Model"] + list(target_names.keys()),
    tabular_data=[
        ["r2"] + metrics["r2"],
        ["mae"] + metrics["mae"],
        ["nmae"] + metrics["nmae"],
        ["pcc"] + metrics["pcc"],
    ],
    floatfmt=".4f",
)
print(table)


plot_parity_plots(truth, pred, target_names, save_path='parity_plots.png')


