import datetime
import os
import numpy as np
from time import perf_counter
from typing import List, Literal, Optional, OrderedDict, Tuple
import adabound
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import distributed
import torch.nn as nn
import torch.nn.functional as F
from fastprop.data import fastpropDataLoader, inverse_standard_scale, standard_scale
from fastprop.defaults import init_logger
from fastprop.metrics import SCORE_LOOKUP

logger = init_logger(__name__)


class ClampN(torch.nn.Module):
    def __init__(self, n: float) -> None:
        super().__init__()
        self.n = n

    def forward(self, batch: torch.Tensor):
        return torch.clamp(batch, min=-self.n, max=self.n)

    def extra_repr(self) -> str:
        return f"n={self.n}"


class FeatureAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=-1)  # 特征维度注意力权重
        )

    def forward(self, x):
        # x: [batch_size, feature_dim]
        weights = self.attention(x)  # shape: [batch_size, feature_dim]
        return x * weights  # 逐特征加权


class fastprop(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 191,
        hidden_size: int = 1800,
        readout_size: int = 1,
        num_tasks: int = 1,
        learning_rate: float = 0.001,
        fnn_layers: int = 2,
        clamp_input: bool = False,
        problem_type: Literal["regression", "binary", "multiclass", "multilabel"] = "regression",
        target_names: List[str] = [],
        feature_means: Optional[torch.Tensor] = None,
        feature_vars: Optional[torch.Tensor] = None,
        target_means: Optional[torch.Tensor] = None,
        target_vars: Optional[torch.Tensor] = None,
    ):
        """fastprop descriptor-based regression FNN.

        Args:
            input_size (int, optional): Number of features. Defaults to 1613.
            hidden_size (int, optional): Height of hidden layers. Defaults to 1800.
            readout_size (int, optional): Number of output neurons. Defaults to 1.
            num_tasks (int, optional): Number of distinct tasks. Defaults to 1.
            learning_rate (float, optional): Learning rate for SGD. Defaults to 0.001.
            fnn_layers (int, optional): Number of hidden layers. Defaults to 2.
            clamp_input (bool, optional): Clamp inputs to +/-3 to aid extrapolation. Defaults to False.
            problem_type (Literal["regression", "binary", "multiclass", "multilabel"], optional): Type of training task. Defaults to "regression".
            target_names (list[str], optional): Names for targets in dataset, blank for simple integer names. Defaults to [].
            feature_means (Optional[torch.Tensor], optional): Means for scaling features in regression. Defaults to None.
            feature_vars (Optional[torch.Tensor], optional): Variances for scaling features in regression. Defaults to None.
            target_means (Optional[torch.Tensor], optional): Means for scaling targets during inference. Defaults to None.
            target_vars (Optional[torch.Tensor], optional): Means for scaling targets during inference. Defaults to None.
        """
        super().__init__()
        self.n_tasks = num_tasks
        self.register_buffer("feature_means", feature_means)
        self.register_buffer("feature_vars", feature_vars)
        self.register_buffer("target_means", target_means)
        self.register_buffer("target_vars", target_vars)
        self.problem_type = problem_type
        self.training_metric = fastprop.get_metric(problem_type)
        self.learning_rate = learning_rate
        self.target_names = target_names

        # fully-connected nn
        layers = OrderedDict()
        if clamp_input:
            layers["clamp"] = ClampN(3)
        for i in range(fnn_layers):
            layers[f"lin{i+1}"] = torch.nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            if fnn_layers == 1 or i < (fnn_layers - 1):  # no output activation, unless single layer
                layers[f"act{i+1}"] = torch.nn.ReLU()
        self.fnn = torch.nn.Sequential(layers)
        self.readout = torch.nn.Linear(hidden_size, readout_size)

        self.save_hyperparameters()

    # def configure_optimizers(self):
    #     """See https://lightning.ai/docs/pytorch/stable/common/optimization.html
    #
    #     Returns:
    #         dict: Optimizer name and instance.
    #     """
    #     return {"optimizer": torch.optim.Adam(self.parameters(), lr=self.learning_rate)}
    def configure_optimizers(self):
        import adabound  # 导入AdaBound库
        return {"optimizer": adabound.AdaBound(
            self.parameters(),
            lr=self.learning_rate,  # 初始学习率（建议0.001）
            final_lr=0.005,  # 最终学习率（关键参数！）
            betas=(0.9, 0.999),  # 动量参数（同Adam）
            eps=1e-8  # 数值稳定性
        )}

    def setup(self, stage=None):
        """Fill the target names if none were provided

        Args:
            stage (str, optional): Step of pipeline. Defaults to None.
        """
        if stage == "fit":
            if len(self.target_names) == 0:
                self.target_names = [f"task_{i}" for i in range(self.n_tasks)]

    def get_metric(problem_type: str):
        """Get the metric for training and early stopping based on the problem type.

        Args:
            problem_type (str): Regression, multilabel, multiclass, or binary.

        Raises:
            RuntimeError: Unsupported problem types

        Returns:
            str: names for the two metrics
        """
        if problem_type == "regression":
            return "mse"
        elif problem_type in {"multilabel", "binary"}:
            return "bce"
        elif problem_type == "multiclass":
            return "kldiv"
        else:
            raise RuntimeError(f"Unsupported problem type '{problem_type}'!")

    def forward(self, x):
        """Returns the logits (i.e. without activation or scaling) for a given batch of features.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Logits.
        """
        x = self.fnn.forward(x)
        x = self.readout(x)
        # print(f"模型输出维度: {x.shape}")
        return x

    def log(self, name, value, **kwargs):
        """Wrap the parent PyTorch Lightning log function to automatically detect DDP."""
        return super().log(name, value, sync_dist=distributed.is_initialized(), **kwargs)

    def training_step(self, batch, batch_idx):
        loss, _ = self._machine_loss(batch)
        self.log(f"train_{self.training_metric}_scaled_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self._machine_loss(batch)
        self.log(f"validation_{self.training_metric}_scaled_loss", loss)
        self._human_loss(y_hat, batch, "validation")
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat = self._machine_loss(batch)
        self.log(f"test_{self.training_metric}_scaled_loss", loss)
        self._human_loss(y_hat, batch, "test")
        return loss

    def predict_step(self, batch: Tuple[torch.Tensor]):
        """Applies feature scaling and appropriate activation function to a Tensor of descriptors.

        Args:
            batch (tuple[torch.Tensor]): Unscaled descriptors.

        Returns:
            torch.Tensor: Predictions.
        """
        descriptors = batch[0]
        if self.feature_means is not None and self.feature_vars is not None:
            descriptors = standard_scale(descriptors, self.feature_means, self.feature_vars)
        with torch.inference_mode():
            logits = self.forward(descriptors)
        if self.problem_type == "regression":
            logits = inverse_standard_scale(logits, self.target_means, self.target_vars)
            return logits
        elif self.problem_type in {"multilabel", "binary"}:
            return torch.sigmoid(logits)
        elif self.problem_type == "multiclass":
            return torch.nn.functional.softmax(logits, dim=1)

    def _machine_loss(self, batch):
        """Reports the loss without undoing scaling for greater efficiency."""
        x, y = batch
        y_hat = self.forward(x)
        if self.problem_type == "regression":
            loss = torch.nn.functional.mse_loss(y_hat, y, reduction="mean")
        elif self.problem_type in {"multilabel", "binary"}:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y, reduction="mean")
        else:
            y_pred = torch.softmax(y_hat, dim=1)
            loss = torch.nn.functional.kl_div(y_pred.log(), y.float(), reduction="batchmean")
        return loss, y_hat

    def _human_loss(self, pred, batch, name):
        """Reports human-interpretable loss metrics.

        Args:
            pred (torch.Tensor): Network logits.
            batch (tuple[torch.Tensor, torch.Tensor]): Inputs and targets
            name (str): Name under which to log the results.
        """
        truth = batch[1]
        # 样本数量检查
        if truth.size(0) < 2:
            print(f"Skip metric calculation for batch size {truth.size(0)}")
            return

        if self.problem_type == "regression" and self.target_means is not None and self.target_vars is not None:
            pred = inverse_standard_scale(pred, self.target_means, self.target_vars)
            truth = inverse_standard_scale(truth, self.target_means, self.target_vars)
        # get probability from logits
        elif self.problem_type in {"binary", "multilabel"}:
            pred = torch.sigmoid(pred)
        else:
            pred = torch.nn.functional.softmax(pred, dim=1)

        for metric in SCORE_LOOKUP[self.problem_type]:
            self.log(f"{name}_{metric.__name__}", metric(truth, pred, self.readout.out_features))
            if "multi" not in self.problem_type and self.n_tasks > 1:
                per_task_metric = metric(truth, pred, None, True)
                for target, value in zip(self.target_names, per_task_metric):
                    self.log(f"{name}_{target}_{metric.__name__}", value)

class fastprop_up(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 1613,
        hidden_size: int = 1800,
        readout_size: int = 1,
        num_tasks: int = 1,
        learning_rate: float = 0.001,
        fnn_layers: int = 2,
        clamp_input: bool = False,
        problem_type: Literal["regression", "binary", "multiclass", "multilabel"] = "regression",
        target_names: List[str] = [],
        feature_means: Optional[torch.Tensor] = None,
        feature_vars: Optional[torch.Tensor] = None,
        target_means: Optional[torch.Tensor] = None,
        target_vars: Optional[torch.Tensor] = None,
    ):
        """fastprop descriptor-based regression FNN.

        Args:
            input_size (int, optional): Number of features. Defaults to 1613.
            hidden_size (int, optional): Height of hidden layers. Defaults to 1800.
            readout_size (int, optional): Number of output neurons. Defaults to 1.
            num_tasks (int, optional): Number of distinct tasks. Defaults to 1.
            learning_rate (float, optional): Learning rate for SGD. Defaults to 0.001.
            fnn_layers (int, optional): Number of hidden layers. Defaults to 2.
            clamp_input (bool, optional): Clamp inputs to +/-3 to aid extrapolation. Defaults to False.
            problem_type (Literal["regression", "binary", "multiclass", "multilabel"], optional): Type of training task. Defaults to "regression".
            target_names (list[str], optional): Names for targets in dataset, blank for simple integer names. Defaults to [].
            feature_means (Optional[torch.Tensor], optional): Means for scaling features in regression. Defaults to None.
            feature_vars (Optional[torch.Tensor], optional): Variances for scaling features in regression. Defaults to None.
            target_means (Optional[torch.Tensor], optional): Means for scaling targets during inference. Defaults to None.
            target_vars (Optional[torch.Tensor], optional): Means for scaling targets during inference. Defaults to None.
        """
        super().__init__()
        self.n_tasks = num_tasks
        self.register_buffer("feature_means", feature_means)
        self.register_buffer("feature_vars", feature_vars)
        self.register_buffer("target_means", target_means)
        self.register_buffer("target_vars", target_vars)
        self.problem_type = problem_type
        self.training_metric = fastprop.get_metric(problem_type)
        self.learning_rate = learning_rate
        self.target_names = target_names

        # fully-connected nn
        layers = OrderedDict()
        if clamp_input:
            layers["clamp"] = ClampN(3)
        for i in range(fnn_layers):
            layers[f"lin{i+1}"] = torch.nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            if fnn_layers == 1 or i < (fnn_layers - 1):  # no output activation, unless single layer
                layers[f"act{i+1}"] = torch.nn.ReLU()
        self.fnn = torch.nn.Sequential(layers)
        self.readout = torch.nn.Linear(hidden_size, readout_size)

        self.save_hyperparameters()

    # def configure_optimizers(self):
    #     """See https://lightning.ai/docs/pytorch/stable/common/optimization.html
    #
    #     Returns:
    #         dict: Optimizer name and instance.
    #     """
    #     return {"optimizer": torch.optim.Adam(self.parameters(), lr=self.learning_rate)}

    # def configure_optimizers(self):
    #     optimizer = adabound.AdaBound(
    #         self.parameters(),
    #         lr=self.learning_rate,
    #         final_lr=0.01,  # 你可以根据需要调整 final_lr
    #         weight_decay=1e-4,  # 可设置为 1e-5 等进行正则
    #         amsbound=True  # 若想增加收敛稳定性可设为 True
    #     )

    def configure_optimizers(self):
        optimizer = adabound.AdaBound(
            self.parameters(),
            lr=self.learning_rate,
            final_lr=0.01,
            weight_decay=1e-4,
            amsbound=True
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]

    def setup(self, stage=None):
        """Fill the target names if none were provided

        Args:
            stage (str, optional): Step of pipeline. Defaults to None.
        """
        if stage == "fit":
            if len(self.target_names) == 0:
                self.target_names = [f"task_{i}" for i in range(self.n_tasks)]

    def get_metric(problem_type: str):
        """Get the metric for training and early stopping based on the problem type.

        Args:
            problem_type (str): Regression, multilabel, multiclass, or binary.

        Raises:
            RuntimeError: Unsupported problem types

        Returns:
            str: names for the two metrics
        """
        if problem_type == "regression":
            return "mse"
        elif problem_type in {"multilabel", "binary"}:
            return "bce"
        elif problem_type == "multiclass":
            return "kldiv"
        else:
            raise RuntimeError(f"Unsupported problem type '{problem_type}'!")

    def forward(self, x):
        """Returns the logits (i.e. without activation or scaling) for a given batch of features.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Logits.
        """
        x = self.fnn.forward(x)
        x = self.readout(x)
        print(f"模型输出维度: {x.shape}")
        return x

    def log(self, name, value, **kwargs):
        """Wrap the parent PyTorch Lightning log function to automatically detect DDP."""
        return super().log(name, value, sync_dist=distributed.is_initialized(), **kwargs)

    def training_step(self, batch, batch_idx):
        loss, _ = self._machine_loss(batch)
        self.log(f"train_{self.training_metric}_scaled_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self._machine_loss(batch)
        self.log(f"validation_{self.training_metric}_scaled_loss", loss)
        self._human_loss(y_hat, batch, "validation")
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat = self._machine_loss(batch)
        self.log(f"test_{self.training_metric}_scaled_loss", loss)
        self._human_loss(y_hat, batch, "test")
        return loss

    def predict_step(self, batch: Tuple[torch.Tensor]):
        """Applies feature scaling and appropriate activation function to a Tensor of descriptors.

        Args:
            batch (tuple[torch.Tensor]): Unscaled descriptors.

        Returns:
            torch.Tensor: Predictions.
        """
        descriptors = batch[0]
        if self.feature_means is not None and self.feature_vars is not None:
            descriptors = standard_scale(descriptors, self.feature_means, self.feature_vars)
        with torch.inference_mode():
            logits = self.forward(descriptors)
        if self.problem_type == "regression":
            logits = inverse_standard_scale(logits, self.target_means, self.target_vars)
            return logits
        elif self.problem_type in {"multilabel", "binary"}:
            return torch.sigmoid(logits)
        elif self.problem_type == "multiclass":
            return torch.nn.functional.softmax(logits, dim=1)

    def _machine_loss(self, batch):
        """Reports the loss without undoing scaling for greater efficiency."""
        x, y = batch
        y_hat = self.forward(x)
        if self.problem_type == "regression":
            loss = torch.nn.functional.mse_loss(y_hat, y, reduction="mean")
        elif self.problem_type in {"multilabel", "binary"}:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y, reduction="mean")
        else:
            y_pred = torch.softmax(y_hat, dim=1)
            loss = torch.nn.functional.kl_div(y_pred.log(), y.float(), reduction="batchmean")
        return loss, y_hat

    def _human_loss(self, pred, batch, name):
        """Reports human-interpretable loss metrics.

        Args:
            pred (torch.Tensor): Network logits.
            batch (tuple[torch.Tensor, torch.Tensor]): Inputs and targets
            name (str): Name under which to log the results.
        """
        truth = batch[1]
        if self.problem_type == "regression" and self.target_means is not None and self.target_vars is not None:
            pred = inverse_standard_scale(pred, self.target_means, self.target_vars)
            truth = inverse_standard_scale(truth, self.target_means, self.target_vars)
        # get probability from logits
        elif self.problem_type in {"binary", "multilabel"}:
            pred = torch.sigmoid(pred)
        else:
            pred = torch.nn.functional.softmax(pred, dim=1)

        for metric in SCORE_LOOKUP[self.problem_type]:
            self.log(f"{name}_{metric.__name__}", metric(truth, pred, self.readout.out_features))
            if "multi" not in self.problem_type and self.n_tasks > 1:
                per_task_metric = metric(truth, pred, None, True)
                for target, value in zip(self.target_names, per_task_metric):
                    self.log(f"{name}_{target}_{metric.__name__}", value)

class MultiTaskRegressor(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 350,
        hidden_size: int = 1800,
        readout_size: int = 4,  # 修改为 num_tasks
        num_tasks: int = 4,
        learning_rate: float = 0.001,
        fnn_layers: int = 2,
        clamp_input: bool = False,
        problem_type: Literal["regression", "binary", "multiclass", "multilabel"] = "regression",
        target_names: List[str] = [],
        feature_means: Optional[torch.Tensor] = None,
        feature_vars: Optional[torch.Tensor] = None,
        target_means: Optional[torch.Tensor] = None,
        target_vars: Optional[torch.Tensor] = None,
        # activity_regularization: float = 0.1,  # 添加L2正则
        # loss_clip_value: float = 10.0,  # 限制损失值范围
    ):
        super().__init__()
        self.n_tasks = num_tasks
        self.register_buffer("feature_means", feature_means)
        self.register_buffer("feature_vars", feature_vars)
        self.register_buffer("target_means", target_means)
        self.register_buffer("target_vars", target_vars)
        self.problem_type = problem_type
        self.training_metric = fastprop.get_metric(problem_type)
        self.learning_rate = learning_rate
        self.target_names = target_names

        # fully-connected nn
        layers = OrderedDict()
        if clamp_input:
            layers["clamp"] = ClampN(3)
        for i in range(fnn_layers):
            layers[f"lin{i+1}"] = torch.nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            if fnn_layers == 1 or i < (fnn_layers - 1):
                layers[f"act{i+1}"] = torch.nn.ReLU()
        self.fnn = torch.nn.Sequential(layers)
        self.readout = torch.nn.Linear(hidden_size, readout_size)  # 修改输出层

        self.save_hyperparameters()

    def configure_optimizers(self):
        """See https://lightning.ai/docs/pytorch/stable/common/optimization.html

        Returns:
            dict: Optimizer name and instance.
        """
        return {"optimizer": torch.optim.Adam(self.parameters(), lr=self.learning_rate)}

    def setup(self, stage=None):
        """Fill the target names if none were provided

        Args:
            stage (str, optional): Step of pipeline. Defaults to None.
        """
        if stage == "fit":
            if len(self.target_names) == 0:
                self.target_names = [f"task_{i}" for i in range(self.n_tasks)]

    def get_metric(problem_type: str):
        """Get the metric for training and early stopping based on the problem type.

        Args:
            problem_type (str): Regression, multilabel, multiclass, or binary.

        Raises:
            RuntimeError: Unsupported problem types

        Returns:
            str: names for the two metrics
        """
        if problem_type == "regression":
            return "mse"
        elif problem_type in {"multilabel", "binary"}:
            return "bce"
        elif problem_type == "multiclass":
            return "kldiv"
        else:
            raise RuntimeError(f"Unsupported problem type '{problem_type}'!")

    def forward(self, x):
        """Returns the logits (i.e. without activation or scaling) for a given batch of features.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Logits.
        """
        x = self.fnn.forward(x)
        x = self.readout(x)
        # print(f"模型输出维度: {x.shape}")
        return x

    def log(self, name, value, **kwargs):
        """Wrap the parent PyTorch Lightning log function to automatically detect DDP."""
        return super().log(name, value, sync_dist=distributed.is_initialized(), **kwargs)

    def training_step(self, batch, batch_idx):
        loss, _ = self._machine_loss(batch)
        self.log(f"train_{self.training_metric}_scaled_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self._machine_loss(batch)
        self.log(f"validation_{self.training_metric}_scaled_loss", loss)
        self._human_loss(y_hat, batch, "validation")
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat = self._machine_loss(batch)
        self.log(f"test_{self.training_metric}_scaled_loss", loss)
        self._human_loss(y_hat, batch, "test")
        return loss

    def predict_step(self, batch: Tuple[torch.Tensor]):
        """Applies feature scaling and appropriate activation function to a Tensor of descriptors.

        Args:
            batch (tuple[torch.Tensor]): Unscaled descriptors.

        Returns:
            torch.Tensor: Predictions.
        """
        descriptors = batch[0]
        if self.feature_means is not None and self.feature_vars is not None:
            descriptors = standard_scale(descriptors, self.feature_means, self.feature_vars)
        with torch.inference_mode():
            logits = self.forward(descriptors)
        if self.problem_type == "regression":
            logits = inverse_standard_scale(logits, self.target_means, self.target_vars)
            return logits
        elif self.problem_type in {"multilabel", "binary"}:
            return torch.sigmoid(logits)
        elif self.problem_type == "multiclass":
            return torch.nn.functional.softmax(logits, dim=1)

    def _machine_loss(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = 0
        if self.problem_type == "regression":
            for i in range(self.n_tasks):
                task_loss = torch.nn.functional.mse_loss(y_hat[:, i], y[:, i], reduction="mean")
                loss += task_loss
        elif self.problem_type in {"multilabel", "binary"}:
            for i in range(self.n_tasks):
                task_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat[:, i], y[:, i], reduction="mean")
                loss += task_loss
        else:
            for i in range(self.n_tasks):
                y_pred = torch.softmax(y_hat[:, i], dim=1)
                task_loss = torch.nn.functional.kl_div(y_pred.log(), y.float(), reduction="batchmean")
                loss += task_loss
        return loss / self.n_tasks, y_hat  # 返回平均损失

    def _human_loss(self, pred, batch, name):
        """Reports human-interpretable loss metrics.

        Args:
            pred (torch.Tensor): Network logits.
            batch (tuple[torch.Tensor, torch.Tensor]): Inputs and targets
            name (str): Name under which to log the results.
        """
        truth = batch[1]
        if self.problem_type == "regression" and self.target_means is not None and self.target_vars is not None:
            pred = inverse_standard_scale(pred, self.target_means, self.target_vars)
            truth = inverse_standard_scale(truth, self.target_means, self.target_vars)
        # get probability from logits
        elif self.problem_type in {"binary", "multilabel"}:
            pred = torch.sigmoid(pred)
        else:
            pred = torch.nn.functional.softmax(pred, dim=1)

        for metric in SCORE_LOOKUP[self.problem_type]:
            scores = metric(truth, pred, self.readout.out_features)

            # 判断scores是否是单个标量（numpy float, python float, 或0维tensor）
            if isinstance(scores, (float, np.floating)):
                # numpy或python浮点数，直接记录
                self.log(f"{name}_{metric.__name__}_task0", scores)
            elif isinstance(scores, torch.Tensor):
                if scores.ndim == 0:
                    # 0维tensor标量
                    self.log(f"{name}_{metric.__name__}_task0", scores.item())
                else:
                    # 多任务tensor，逐个记录
                    for i, score in enumerate(scores):
                        self.log(f"{name}_{metric.__name__}_task{i}", score.item())
            elif isinstance(scores, (list, tuple, np.ndarray)):
                # 多任务list/array，逐个记录
                for i, score in enumerate(scores):
                    # 如果是numpy类型转成python float
                    if isinstance(score, np.generic):
                        score = score.item()
                    self.log(f"{name}_{metric.__name__}_task{i}", score)
            else:
                # 不支持的类型，强制转换为字符串记录，避免崩溃
                self.log(f"{name}_{metric.__name__}_task0", str(scores))

            if "multi" not in self.problem_type and self.n_tasks > 1:
                per_task_metric = metric(truth, pred, None, True)
                for target, value in zip(self.target_names, per_task_metric):
                    self.log(f"{name}_{target}_{metric.__name__}", value)

class MultiTask_PINNloss(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 737,
        hidden_size: int = 1800,
        readout_size: int = 4,  # 修改为 num_tasks
        num_tasks: int = 4,
        learning_rate: float = 0.001,
        fnn_layers: int = 2,
        clamp_input: bool = False,
        problem_type: Literal["regression", "binary", "multiclass", "multilabel"] = "regression",
        target_names: List[str] = [],
        feature_means: Optional[torch.Tensor] = None,
        feature_vars: Optional[torch.Tensor] = None,
        target_means: Optional[torch.Tensor] = None,
        target_vars: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_tasks = num_tasks
        self.register_buffer("feature_means", feature_means)
        self.register_buffer("feature_vars", feature_vars)
        self.register_buffer("target_means", target_means)
        self.register_buffer("target_vars", target_vars)
        self.problem_type = problem_type
        self.training_metric = fastprop.get_metric(problem_type)
        self.learning_rate = learning_rate
        self.target_names = target_names
        # PINNloss 新增参数
        self.lambda_SH = 0.4  # L_S-H 权重
        self.lambda_comb = 0.3  # L_comb 权重
        self.lambda_expl = 0.3  # L_expl 权重

        # fully-connected nn
        layers = OrderedDict()
        if clamp_input:
            layers["clamp"] = ClampN(3)
        for i in range(fnn_layers):
            layers[f"lin{i+1}"] = torch.nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            if fnn_layers == 1 or i < (fnn_layers - 1):
                layers[f"act{i+1}"] = torch.nn.ReLU()
        self.fnn = torch.nn.Sequential(layers)
        self.readout = torch.nn.Linear(hidden_size, readout_size)  # 修改输出层

        self.save_hyperparameters()

    def configure_optimizers(self):
        """See https://lightning.ai/docs/pytorch/stable/common/optimization.html

        Returns:
            dict: Optimizer name and instance.
        """
        return {"optimizer": torch.optim.Adam(self.parameters(), lr=self.learning_rate)}

    def setup(self, stage=None):
        """Fill the target names if none were provided

        Args:
            stage (str, optional): Step of pipeline. Defaults to None.
        """
        if stage == "fit":
            if len(self.target_names) == 0:
                self.target_names = [f"task_{i}" for i in range(self.n_tasks)]

    def get_metric(problem_type: str):
        """Get the metric for training and early stopping based on the problem type.

        Args:
            problem_type (str): Regression, multilabel, multiclass, or binary.

        Raises:
            RuntimeError: Unsupported problem types

        Returns:
            str: names for the two metrics
        """
        if problem_type == "regression":
            return "mse"
        elif problem_type in {"multilabel", "binary"}:
            return "bce"
        elif problem_type == "multiclass":
            return "kldiv"
        else:
            raise RuntimeError(f"Unsupported problem type '{problem_type}'!")

    def forward(self, x):
        """Returns the logits (i.e. without activation or scaling) for a given batch of features.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Logits.
        """
        x = self.fnn.forward(x)
        x = self.readout(x)
        # print(f"模型输出维度: {x.shape}")
        return x

    def log(self, name, value, **kwargs):
        """Wrap the parent PyTorch Lightning log function to automatically detect DDP."""
        return super().log(name, value, sync_dist=distributed.is_initialized(), **kwargs)

    def training_step(self, batch, batch_idx):
        loss, _ = self._machine_loss(batch)
        self.log(f"train_{self.training_metric}_scaled_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self._machine_loss(batch)
        self.log(f"validation_{self.training_metric}_scaled_loss", loss)
        self._human_loss(y_hat, batch, "validation")
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat = self._machine_loss(batch)
        self.log(f"test_{self.training_metric}_scaled_loss", loss)
        self._human_loss(y_hat, batch, "test")
        return loss

    def predict_step(self, batch: Tuple[torch.Tensor]):
        """Applies feature scaling and appropriate activation function to a Tensor of descriptors.

        Args:
            batch (tuple[torch.Tensor]): Unscaled descriptors.

        Returns:
            torch.Tensor: Predictions.
        """
        descriptors = batch[0]
        if self.feature_means is not None and self.feature_vars is not None:
            descriptors = standard_scale(descriptors, self.feature_means, self.feature_vars)
        with torch.inference_mode():
            logits = self.forward(descriptors)
        if self.problem_type == "regression":
            logits = inverse_standard_scale(logits, self.target_means, self.target_vars)
            return logits
        elif self.problem_type in {"multilabel", "binary"}:
            return torch.sigmoid(logits)
        elif self.problem_type == "multiclass":
            return torch.nn.functional.softmax(logits, dim=1)

    # 新增的各种物理损失函数
    def physics_loss_SH(self, pred: torch.Tensor, temperature: float = 298.15):
        H = pred[:, self.target_names.index("Enthalpy")]  # 焓
        S = pred[:, self.target_names.index("Entropy")]  # 熵
        return torch.mean((H - temperature * S) ** 2)

    def physics_loss_comb(self, pred: torch.Tensor, alpha=1.0, beta=0.0):
        Q_comb = pred[:, self.target_names.index("Heat_comb")]
        H = pred[:, self.target_names.index("Enthalpy")]
        return torch.mean((Q_comb - (alpha * H + beta)) ** 2)

    def physics_loss_expl(self, pred: torch.Tensor, w1=1.0, w2=1.0, w3=1.0, c=0.0):
        Q_expl = pred[:, self.target_names.index("Heat_expl")]
        H = pred[:, self.target_names.index("Enthalpy")]
        S = pred[:, self.target_names.index("Entropy")]
        Q_comb = pred[:, self.target_names.index("Heat_comb")]
        return torch.mean((Q_expl - (w1 * H + w2 * S + w3 * Q_comb + c)) ** 2)

    def _machine_loss(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = 0
        if self.problem_type == "regression":
            for i in range(self.n_tasks):
                task_loss = torch.nn.functional.mse_loss(y_hat[:, i], y[:, i], reduction="mean")
                loss += task_loss
            loss = loss / self.n_tasks  # L_data

            if all(name in self.target_names for name in ["enthalpy", "entropy", "combustion_heat", "explosion_heat"]):
                L_physics_SH = self.physics_loss_SH(y_hat)
                L_physics_comb = self.physics_loss_comb(y_hat)
                L_physics_expl = self.physics_loss_expl(y_hat)
                L_physics = (
                        self.lambda_SH * L_physics_SH +
                        self.lambda_comb * L_physics_comb +
                        self.lambda_expl * L_physics_expl
                )
                loss += L_physics  # L_total = L_data + L_physics
        # elif self.problem_type in {"multilabel", "binary"}:
        #     for i in range(self.n_tasks):
        #         task_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat[:, i], y[:, i], reduction="mean")
        #         loss += task_loss
        # else:
        #     for i in range(self.n_tasks):
        #         y_pred = torch.softmax(y_hat[:, i], dim=1)
        #         task_loss = torch.nn.functional.kl_div(y_pred.log(), y.float(), reduction="batchmean")
        #         loss += task_loss
        return loss, y_hat  # 返回平均损失

    def _human_loss(self, pred, batch, name):
        """Reports human-interpretable loss metrics.

        Args:
            pred (torch.Tensor): Network logits.
            batch (tuple[torch.Tensor, torch.Tensor]): Inputs and targets
            name (str): Name under which to log the results.
        """
        truth = batch[1]
        if self.problem_type == "regression" and self.target_means is not None and self.target_vars is not None:
            pred = inverse_standard_scale(pred, self.target_means, self.target_vars)
            truth = inverse_standard_scale(truth, self.target_means, self.target_vars)
        # get probability from logits
        elif self.problem_type in {"binary", "multilabel"}:
            pred = torch.sigmoid(pred)
        else:
            pred = torch.nn.functional.softmax(pred, dim=1)

        for metric in SCORE_LOOKUP[self.problem_type]:
            scores = metric(truth, pred, self.readout.out_features)

            # 判断scores是否是单个标量（numpy float, python float, 或0维tensor）
            if isinstance(scores, (float, np.floating)):
                # numpy或python浮点数，直接记录
                self.log(f"{name}_{metric.__name__}_task0", scores)
            elif isinstance(scores, torch.Tensor):
                if scores.ndim == 0:
                    # 0维tensor标量
                    self.log(f"{name}_{metric.__name__}_task0", scores.item())
                else:
                    # 多任务tensor，逐个记录
                    for i, score in enumerate(scores):
                        self.log(f"{name}_{metric.__name__}_task{i}", score.item())
            elif isinstance(scores, (list, tuple, np.ndarray)):
                # 多任务list/array，逐个记录
                for i, score in enumerate(scores):
                    # 如果是numpy类型转成python float
                    if isinstance(score, np.generic):
                        score = score.item()
                    self.log(f"{name}_{metric.__name__}_task{i}", score)
            else:
                # 不支持的类型，强制转换为字符串记录，避免崩溃
                self.log(f"{name}_{metric.__name__}_task0", str(scores))

            if "multi" not in self.problem_type and self.n_tasks > 1:
                per_task_metric = metric(truth, pred, None, True)
                for target, value in zip(self.target_names, per_task_metric):
                    self.log(f"{name}_{target}_{metric.__name__}", value)


class MultiTask_PINN_config(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 122,
        hidden_size: int = 1800,
        readout_size: int = 4,  # 修改为 num_tasks
        num_tasks: int = 1,
        learning_rate: float = 0.0001,
        fnn_layers: int = 2,
        clamp_input: bool = False,
        problem_type: Literal["regression", "binary", "multiclass", "multilabel"] = "regression",
        target_names: List[str] = [],
        num_samples: int = 300,
        feature_means: Optional[torch.Tensor] = None,
        feature_vars: Optional[torch.Tensor] = None,
        target_means: Optional[torch.Tensor] = None,
        target_vars: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_tasks = num_tasks
        self.register_buffer("feature_means", feature_means)
        self.register_buffer("feature_vars", feature_vars)
        self.register_buffer("target_means", target_means)
        self.register_buffer("target_vars", target_vars)
        self.problem_type = problem_type
        self.training_metric = fastprop.get_metric(problem_type)
        self.learning_rate = learning_rate
        self.target_names = target_names
        # 优化器新增参数
        self.num_samples=num_samples,
        # PINNloss 新增参数
        self.lambda_SH = 0.4  # L_S-H 权重
        self.lambda_comb = 0.3  # L_comb 权重
        self.lambda_expl = 0.3  # L_expl 权重

        # fully-connected nn
        layers = OrderedDict()
        if clamp_input:
            layers["clamp"] = ClampN(3)
        for i in range(fnn_layers):
            layers[f"lin{i+1}"] = torch.nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            if fnn_layers == 1 or i < (fnn_layers - 1):
                layers[f"act{i+1}"] = torch.nn.ReLU()
                # layers[f"act{i+1}"] = torch.nn.LeakyReLU(negative_slope=0.01)  # 默认负斜率为0.01
                # layers[f"act{i + 1}"] = torch.nn.Tanh()
                # layers[f"act{i + 1}"] = torch.nn.PReLU(init=0.25)
        self.fnn = torch.nn.Sequential(layers)
        self.readout = torch.nn.Linear(hidden_size, readout_size)  # 修改输出层

        self.save_hyperparameters()

    # def configure_optimizers(self):
    #     """See https://lightning.ai/docs/pytorch/stable/common/optimization.html
    #
    #     Returns:
    #         dict: Optimizer name and instance.
    #     """
    #     return {"optimizer": torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))}

    def configure_optimizers(self):
        import adabound  # 导入AdaBound库
        return {"optimizer": adabound.AdaBound(
            self.parameters(),
            lr=self.learning_rate,  # 初始学习率（建议0.001）
            final_lr=0.005,  # 最终学习率（关键参数！）
            betas=(0.9, 0.999),  # 动量参数（同Adam）
            eps=1e-8  # 数值稳定性
        )}

    def setup(self, stage=None):
        """Fill the target names if none were provided

        Args:
            stage (str, optional): Step of pipeline. Defaults to None.
        """
        if stage == "fit":
            if len(self.target_names) == 0:
                self.target_names = [f"task_{i}" for i in range(self.n_tasks)]

    def get_metric(problem_type: str):
        """Get the metric for training and early stopping based on the problem type.

        Args:
            problem_type (str): Regression, multilabel, multiclass, or binary.

        Raises:
            RuntimeError: Unsupported problem types

        Returns:
            str: names for the two metrics
        """
        if problem_type == "regression":
            return "mse"
        elif problem_type in {"multilabel", "binary"}:
            return "bce"
        elif problem_type == "multiclass":
            return "kldiv"
        else:
            raise RuntimeError(f"Unsupported problem type '{problem_type}'!")

    def forward(self, x):
        """Returns the logits (i.e. without activation or scaling) for a given batch of features.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Logits.
        """
        x = self.fnn.forward(x)
        x = self.readout(x)
        # print(f"模型输出维度: {x.shape}")
        return x

    def log(self, name, value, **kwargs):
        """Wrap the parent PyTorch Lightning log function to automatically detect DDP."""
        return super().log(name, value, sync_dist=distributed.is_initialized(), **kwargs)

    def training_step(self, batch, batch_idx):
        loss, _ = self._machine_loss(batch)
        self.log(f"train_{self.training_metric}_scaled_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self._machine_loss(batch)
        self.log(f"validation_{self.training_metric}_scaled_loss", loss)
        self._human_loss(y_hat, batch, "validation")
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat = self._machine_loss(batch)
        self.log(f"test_{self.training_metric}_scaled_loss", loss)
        self._human_loss(y_hat, batch, "test")
        return loss

    def predict_step(self, batch: Tuple[torch.Tensor]):
        """Applies feature scaling and appropriate activation function to a Tensor of descriptors.

        Args:
            batch (tuple[torch.Tensor]): Unscaled descriptors.

        Returns:
            torch.Tensor: Predictions.
        """
        descriptors = batch[0]
        if self.feature_means is not None and self.feature_vars is not None:
            descriptors = standard_scale(descriptors, self.feature_means, self.feature_vars)
        with torch.inference_mode():
            logits = self.forward(descriptors)
        if self.problem_type == "regression":
            logits = inverse_standard_scale(logits, self.target_means, self.target_vars)
            return logits
        elif self.problem_type in {"multilabel", "binary"}:
            return torch.sigmoid(logits)
        elif self.problem_type == "multiclass":
            return torch.nn.functional.softmax(logits, dim=1)

    # 新增的各种物理损失函数
    def physics_loss_SH(self, pred: torch.Tensor, temperature: float = 298.15):
        H = pred[:, self.target_names.index("Enthalpy")]  # 焓
        S = pred[:, self.target_names.index("Entropy")]  # 熵
        return torch.mean((H - temperature * S) ** 2)

    def physics_loss_comb(self, pred: torch.Tensor, alpha=1.0, beta=0.0):
        Q_comb = pred[:, self.target_names.index("Heat_comb")]
        H = pred[:, self.target_names.index("Enthalpy")]
        return torch.mean((Q_comb - (alpha * H + beta)) ** 2)

    def physics_loss_expl(self, pred: torch.Tensor, w1=1.0, w2=1.0, w3=1.0, c=0.0):
        Q_expl = pred[:, self.target_names.index("Heat_expl")]
        H = pred[:, self.target_names.index("Enthalpy")]
        S = pred[:, self.target_names.index("Entropy")]
        Q_comb = pred[:, self.target_names.index("Heat_comb")]
        return torch.mean((Q_expl - (w1 * H + w2 * S + w3 * Q_comb + c)) ** 2)

    def _machine_loss(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = 0
        if self.problem_type == "regression":
            for i in range(self.n_tasks):
                task_loss = torch.nn.functional.mse_loss(y_hat[:, i], y[:, i], reduction="mean")
                loss += task_loss
            loss = loss / self.n_tasks  # L_data

            if all(name in self.target_names for name in ["enthalpy", "entropy", "combustion_heat", "explosion_heat"]):
                L_physics_SH = self.physics_loss_SH(y_hat)
                L_physics_comb = self.physics_loss_comb(y_hat)
                L_physics_expl = self.physics_loss_expl(y_hat)
                L_physics = (
                        self.lambda_SH * L_physics_SH +
                        self.lambda_comb * L_physics_comb +
                        self.lambda_expl * L_physics_expl
                )
                loss += L_physics  # L_total = L_data + L_physics
        # elif self.problem_type in {"multilabel", "binary"}:
        #     for i in range(self.n_tasks):
        #         task_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat[:, i], y[:, i], reduction="mean")
        #         loss += task_loss
        # else:
        #     for i in range(self.n_tasks):
        #         y_pred = torch.softmax(y_hat[:, i], dim=1)
        #         task_loss = torch.nn.functional.kl_div(y_pred.log(), y.float(), reduction="batchmean")
        #         loss += task_loss
        return loss, y_hat  # 返回平均损失

    def _human_loss(self, pred, batch, name):
        """Reports human-interpretable loss metrics.

        Args:
            pred (torch.Tensor): Network logits.
            batch (tuple[torch.Tensor, torch.Tensor]): Inputs and targets
            name (str): Name under which to log the results.
        """
        truth = batch[1]
        if self.problem_type == "regression" and self.target_means is not None and self.target_vars is not None:
            pred = inverse_standard_scale(pred, self.target_means, self.target_vars)
            truth = inverse_standard_scale(truth, self.target_means, self.target_vars)
        # get probability from logits
        elif self.problem_type in {"binary", "multilabel"}:
            pred = torch.sigmoid(pred)
        else:
            pred = torch.nn.functional.softmax(pred, dim=1)

        for metric in SCORE_LOOKUP[self.problem_type]:
            scores = metric(truth, pred, self.readout.out_features)

            # 判断scores是否是单个标量（numpy float, python float, 或0维tensor）
            if isinstance(scores, (float, np.floating)):
                # numpy或python浮点数，直接记录
                self.log(f"{name}_{metric.__name__}_task0", scores)
            elif isinstance(scores, torch.Tensor):
                if scores.ndim == 0:
                    # 0维tensor标量
                    self.log(f"{name}_{metric.__name__}_task0", scores.item())
                else:
                    # 多任务tensor，逐个记录
                    for i, score in enumerate(scores):
                        self.log(f"{name}_{metric.__name__}_task{i}", score.item())
            elif isinstance(scores, (list, tuple, np.ndarray)):
                # 多任务list/array，逐个记录
                for i, score in enumerate(scores):
                    # 如果是numpy类型转成python float
                    if isinstance(score, np.generic):
                        score = score.item()
                    self.log(f"{name}_{metric.__name__}_task{i}", score)
            else:
                # 不支持的类型，强制转换为字符串记录，避免崩溃
                self.log(f"{name}_{metric.__name__}_task0", str(scores))

            if "multi" not in self.problem_type and self.n_tasks > 1:
                per_task_metric = metric(truth, pred, None, True)
                for target, value in zip(self.target_names, per_task_metric):
                    self.log(f"{name}_{target}_{metric.__name__}", value)


def train_and_test(
    output_directory: str,
    fastprop_model: MultiTask_PINNloss,
    train_dataloader: fastpropDataLoader,
    val_dataloader: fastpropDataLoader,
    test_dataloader: fastpropDataLoader,
    number_epochs: int = 30,
    patience: int = 5,
    quiet: bool = False,
    **trainer_kwargs,
):
    """Run a single train/validate and test iteration.

    Args:
        output_directory (str): Filepath to write logs and checkpoints.
        fastprop_model (fastprop): fastprop LightningModule instance.
        train_dataloader (fastpropDataLoader): Training data.
        val_dataloader (fastpropDataLoader): Validation data.
        test_dataloader (fastpropDataLoader): Testing data.
        number_epochs (int, optional): Maximum number of epochs for training. Defaults to 30.
        patience (int, optional): Number of epochs for early stopping. Defaults to 5.
        quiet (bool, optional): Set True to disable some printing. Default to False.
        trainer_kwargs (dict, optional): Additional arguments to pass the the pl.Trainer

    Returns:
        list[dict]: Lightning model output.
    """
    try:
        repetition_number = len(os.listdir(os.path.join(output_directory, "tensorboard_logs"))) + 1
    except FileNotFoundError:
        repetition_number = 1
    tensorboard_logger = TensorBoardLogger(
        output_directory,
        name="tensorboard_logs",
        version=f"repetition_{repetition_number}",
        default_hp_metric=False,
    )

    callbacks = [
        EarlyStopping(
            monitor=f"validation_{fastprop_model.training_metric}_scaled_loss",
            mode="min",
            verbose=False,
            patience=patience,
        ),
        ModelCheckpoint(
            monitor=f"validation_{fastprop_model.training_metric}_scaled_loss",
            dirpath=os.path.join(output_directory, "checkpoints"),
            filename=f"repetition-{repetition_number}" + "-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=number_epochs,
        enable_progress_bar=not quiet,
        enable_model_summary=not quiet,
        logger=tensorboard_logger,
        log_every_n_steps=1,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    t1_start = perf_counter()
    trainer.fit(fastprop_model, train_dataloader, val_dataloader)
    t1_stop = perf_counter()
    logger.info("Elapsed time during training: " + str(datetime.timedelta(seconds=t1_stop - t1_start)))
    ckpt_path = trainer.checkpoint_callback.best_model_path
    logger.info(f"Reloading best model from checkpoint file: {ckpt_path}")
    fastprop_model = fastprop_model.__class__.load_from_checkpoint(ckpt_path)
    validation_results = trainer.validate(fastprop_model, val_dataloader, verbose=False)
    test_results = trainer.test(fastprop_model, test_dataloader, verbose=False)
    validation_results_df = pd.DataFrame.from_records(validation_results, index=("value",))
    logger.info("Displaying validation results for repetition %d:\n%s", repetition_number, validation_results_df.transpose().to_string())
    test_results_df = pd.DataFrame.from_records(test_results, index=("value",))
    logger.info("Displaying validation results for repetition %d:\n%s", repetition_number, test_results_df.transpose().to_string())
    return test_results, validation_results


# with loss
# class MultiTask_PINN_config(pl.LightningModule):
#     def __init__(
#         self,
#         input_size: int = 191,
#         hidden_size: int = 1800,
#         readout_size: int = 4,  # 修改为 num_tasks
#         num_tasks: int = 1,
#         learning_rate: float = 0.0001,
#         fnn_layers: int = 2,
#         clamp_input: bool = False,
#         problem_type: Literal["regression", "binary", "multiclass", "multilabel"] = "regression",
#         target_names: List[str] = [],
#         num_samples: int = 300,
#         feature_means: Optional[torch.Tensor] = None,
#         feature_vars: Optional[torch.Tensor] = None,
#         target_means: Optional[torch.Tensor] = None,
#         target_vars: Optional[torch.Tensor] = None,
#     ):
#         super().__init__()
#         self.n_tasks = num_tasks
#         self.register_buffer("feature_means", feature_means)
#         self.register_buffer("feature_vars", feature_vars)
#         self.register_buffer("target_means", target_means)
#         self.register_buffer("target_vars", target_vars)
#         self.problem_type = problem_type
#         self.training_metric = fastprop.get_metric(problem_type)
#         self.learning_rate = learning_rate
#         self.target_names = target_names
#         # 优化器新增参数
#         self.num_samples = num_samples
#         # PINNloss 新增参数
#         self.lambda_SH = 0.4  # L_S-H 权重
#         self.lambda_comb = 0.3  # L_comb 权重
#         self.lambda_expl = 0.3  # L_expl 权重
#
#         # ========= 新增：记录每个 epoch 的训练 / 验证 loss =========
#         self.train_epoch_losses: List[float] = []
#         self.val_epoch_losses: List[float] = []
#         self._train_loss_sum = 0.0
#         self._train_loss_batches = 0
#         self._val_loss_sum = 0.0
#         self._val_loss_batches = 0
#         # ====================================================
#
#         # fully-connected nn
#         layers = OrderedDict()
#         if clamp_input:
#             layers["clamp"] = ClampN(3)
#         for i in range(fnn_layers):
#             layers[f"lin{i+1}"] = torch.nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
#             if fnn_layers == 1 or i < (fnn_layers - 1):
#                 layers[f"act{i+1}"] = torch.nn.ReLU()
#         self.fnn = torch.nn.Sequential(layers)
#         self.readout = torch.nn.Linear(hidden_size, readout_size)  # 修改输出层
#
#         self.save_hyperparameters()
#
#     def configure_optimizers(self):
#         import adabound  # 导入AdaBound库
#         return {"optimizer": adabound.AdaBound(
#             self.parameters(),
#             lr=self.learning_rate,  # 初始学习率（建议0.001）
#             final_lr=0.005,  # 最终学习率（关键参数！）
#             betas=(0.9, 0.999),  # 动量参数（同Adam）
#             eps=1e-8  # 数值稳定性
#         )}
#
#     def setup(self, stage=None):
#         if stage == "fit":
#             if len(self.target_names) == 0:
#                 self.target_names = [f"task_{i}" for i in range(self.n_tasks)]
#
#     def get_metric(problem_type: str):
#         if problem_type == "regression":
#             return "mse"
#         elif problem_type in {"multilabel", "binary"}:
#             return "bce"
#         elif problem_type == "multiclass":
#             return "kldiv"
#         else:
#             raise RuntimeError(f"Unsupported problem type '{problem_type}'!")
#
#     def forward(self, x):
#         x = self.fnn.forward(x)
#         x = self.readout(x)
#         return x
#
#     def log(self, name, value, **kwargs):
#         return super().log(name, value, sync_dist=distributed.is_initialized(), **kwargs)
#
#     # ========= 新增：epoch 钩子，用于统计平均 loss =========
#     def on_train_epoch_start(self):
#         self._train_loss_sum = 0.0
#         self._train_loss_batches = 0
#
#     def on_train_epoch_end(self):
#         if self._train_loss_batches > 0:
#             avg = self._train_loss_sum / self._train_loss_batches
#             self.train_epoch_losses.append(avg)
#
#     def on_validation_epoch_start(self):
#         self._val_loss_sum = 0.0
#         self._val_loss_batches = 0
#
#     def on_validation_epoch_end(self):
#         if self._val_loss_batches > 0:
#             avg = self._val_loss_sum / self._val_loss_batches
#             self.val_epoch_losses.append(avg)
#     # =====================================================
#
#     def training_step(self, batch, batch_idx):
#         loss, _ = self._machine_loss(batch)
#
#         # 新增：累加 batch loss
#         self._train_loss_sum += loss.detach().cpu().item()
#         self._train_loss_batches += 1
#
#         self.log(f"train_{self.training_metric}_scaled_loss", loss)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         loss, y_hat = self._machine_loss(batch)
#
#         # 新增：累加 batch loss
#         self._val_loss_sum += loss.detach().cpu().item()
#         self._val_loss_batches += 1
#
#         self.log(f"validation_{self.training_metric}_scaled_loss", loss)
#         self._human_loss(y_hat, batch, "validation")
#         return loss
#
#     def test_step(self, batch, batch_idx):
#         loss, y_hat = self._machine_loss(batch)
#         self.log(f"test_{self.training_metric}_scaled_loss", loss)
#         self._human_loss(y_hat, batch, "test")
#         return loss
#
#     def predict_step(self, batch: Tuple[torch.Tensor]):
#         descriptors = batch[0]
#         if self.feature_means is not None and self.feature_vars is not None:
#             descriptors = standard_scale(descriptors, self.feature_means, self.feature_vars)
#         with torch.inference_mode():
#             logits = self.forward(descriptors)
#         if self.problem_type == "regression":
#             logits = inverse_standard_scale(logits, self.target_means, self.target_vars)
#             return logits
#         elif self.problem_type in {"multilabel", "binary"}:
#             return torch.sigmoid(logits)
#         elif self.problem_type == "multiclass":
#             return torch.nn.functional.softmax(logits, dim=1)
#
#     # 下面 physics_loss_*、_machine_loss、_human_loss 保持不变
#     # 这里只贴你原来的版本（略过注释）
#
#     def physics_loss_SH(self, pred: torch.Tensor, temperature: float = 298.15):
#         H = pred[:, self.target_names.index("Enthalpy")]  # 焓
#         S = pred[:, self.target_names.index("Entropy")]  # 熵
#         return torch.mean((H - temperature * S) ** 2)
#
#     def physics_loss_comb(self, pred: torch.Tensor, alpha=1.0, beta=0.0):
#         Q_comb = pred[:, self.target_names.index("Heat_comb")]
#         H = pred[:, self.target_names.index("Enthalpy")]
#         return torch.mean((Q_comb - (alpha * H + beta)) ** 2)
#
#     def physics_loss_expl(self, pred: torch.Tensor, w1=1.0, w2=1.0, w3=1.0, c=0.0):
#         Q_expl = pred[:, self.target_names.index("Heat_expl")]
#         H = pred[:, self.target_names.index("Enthalpy")]
#         S = pred[:, self.target_names.index("Entropy")]
#         Q_comb = pred[:, self.target_names.index("Heat_comb")]
#         return torch.mean((Q_expl - (w1 * H + w2 * S + w3 * Q_comb + c)) ** 2)
#
#     def _machine_loss(self, batch):
#         x, y = batch
#         y_hat = self.forward(x)
#         loss = 0
#         if self.problem_type == "regression":
#             for i in range(self.n_tasks):
#                 task_loss = torch.nn.functional.mse_loss(y_hat[:, i], y[:, i], reduction="mean")
#                 loss += task_loss
#             loss = loss / self.n_tasks  # L_data
#
#             if all(name in self.target_names for name in ["enthalpy", "entropy", "combustion_heat", "explosion_heat"]):
#                 L_physics_SH = self.physics_loss_SH(y_hat)
#                 L_physics_comb = self.physics_loss_comb(y_hat)
#                 L_physics_expl = self.physics_loss_expl(y_hat)
#                 L_physics = (
#                         self.lambda_SH * L_physics_SH +
#                         self.lambda_comb * L_physics_comb +
#                         self.lambda_expl * L_physics_expl
#                 )
#                 loss += L_physics  # L_total = L_data + L_physics
#         return loss, y_hat
#
#     def _human_loss(self, pred, batch, name):
#         truth = batch[1]
#         if self.problem_type == "regression" and self.target_means is not None and self.target_vars is not None:
#             pred = inverse_standard_scale(pred, self.target_means, self.target_vars)
#             truth = inverse_standard_scale(truth, self.target_means, self.target_vars)
#         elif self.problem_type in {"binary", "multilabel"}:
#             pred = torch.sigmoid(pred)
#         else:
#             pred = torch.nn.functional.softmax(pred, dim=1)
#
#         for metric in SCORE_LOOKUP[self.problem_type]:
#             scores = metric(truth, pred, self.readout.out_features)
#             if isinstance(scores, (float, np.floating)):
#                 self.log(f"{name}_{metric.__name__}_task0", scores)
#             elif isinstance(scores, torch.Tensor):
#                 if scores.ndim == 0:
#                     self.log(f"{name}_{metric.__name__}_task0", scores.item())
#                 else:
#                     for i, score in enumerate(scores):
#                         self.log(f"{name}_{metric.__name__}_task{i}", score.item())
#             elif isinstance(scores, (list, tuple, np.ndarray)):
#                 for i, score in enumerate(scores):
#                     if isinstance(score, np.generic):
#                         score = score.item()
#                     self.log(f"{name}_{metric.__name__}_task{i}", score)
#             else:
#                 self.log(f"{name}_{metric.__name__}_task0", str(scores))
#
#             if "multi" not in self.problem_type and self.n_tasks > 1:
#                 per_task_metric = metric(truth, pred, None, True)
#                 for target, value in zip(self.target_names, per_task_metric):
#                     self.log(f"{name}_{target}_{metric.__name__}", value)
#
#
# def train_and_test(
#     output_directory: str,
#     fastprop_model: MultiTask_PINNloss,
#     train_dataloader: fastpropDataLoader,
#     val_dataloader: fastpropDataLoader,
#     test_dataloader: fastpropDataLoader,
#     number_epochs: int = 30,
#     patience: int = 5,
#     quiet: bool = False,
#     **trainer_kwargs,
# ):
#     try:
#         repetition_number = len(os.listdir(os.path.join(output_directory, "tensorboard_logs"))) + 1
#     except FileNotFoundError:
#         repetition_number = 1
#     tensorboard_logger = TensorBoardLogger(
#         output_directory,
#         name="tensorboard_logs",
#         version=f"repetition_{repetition_number}",
#         default_hp_metric=False,
#     )
#
#     callbacks = [
#         EarlyStopping(
#             monitor=f"validation_{fastprop_model.training_metric}_scaled_loss",
#             mode="min",
#             verbose=False,
#             patience=patience,
#         ),
#         ModelCheckpoint(
#             monitor=f"validation_{fastprop_model.training_metric}_scaled_loss",
#             dirpath=os.path.join(output_directory, "checkpoints"),
#             filename=f"repetition-{repetition_number}" + "-{epoch:02d}-{val_loss:.2f}",
#             save_top_k=1,
#             mode="min",
#         ),
#     ]
#
#     trainer = pl.Trainer(
#         max_epochs=number_epochs,
#         enable_progress_bar=not quiet,
#         enable_model_summary=not quiet,
#         logger=tensorboard_logger,
#         log_every_n_steps=1,
#         enable_checkpointing=True,
#         check_val_every_n_epoch=1,
#         callbacks=callbacks,
#         **trainer_kwargs,
#     )
#
#     t1_start = perf_counter()
#     trainer.fit(fastprop_model, train_dataloader, val_dataloader)
#     t1_stop = perf_counter()
#     logger.info("Elapsed time during training: " + str(datetime.timedelta(seconds=t1_stop - t1_start)))
#
#     # ========= 绘制训练 / 验证损失曲线，保存为 loss.svg =========
#     try:
#         train_losses = getattr(fastprop_model, "train_epoch_losses", [])
#         val_losses = getattr(fastprop_model, "val_epoch_losses", [])
#
#         if len(train_losses) > 0:
#             import matplotlib.pyplot as plt
#             import numpy as np
#
#             epochs_train = np.arange(1, len(train_losses) + 1)
#             epochs_val = np.arange(1, len(val_losses) + 1) if len(val_losses) > 0 else None
#
#             plt.figure(figsize=(6, 4), dpi=300)
#             plt.plot(epochs_train, train_losses, label="Training Loss", linewidth=2)
#
#             # 只要有验证 loss，就画出来，不再强制长度相等
#             if epochs_val is not None:
#                 plt.plot(epochs_val, val_losses, label="Validation Loss", linewidth=2)
#
#             plt.xlabel("Epoch", fontsize=14)
#             plt.ylabel("Loss (MSE)", fontsize=14)
#             plt.title("Training vs Validation Loss (qm9_thermo2000)", fontsize=14)
#             plt.legend(fontsize=12)
#             plt.grid(alpha=0.3)
#
#             # —— 关键：为了看清后面的趋势，按第 2 个 epoch 之后的范围自动缩放 y 轴 ——
#             if len(train_losses) > 1:
#                 all_vals = train_losses[1:]
#                 if len(val_losses) > 1:
#                     all_vals = np.concatenate([all_vals, val_losses[1:]])
#                 ymin = 0.0
#                 ymax = float(np.max(all_vals)) * 1.2
#                 plt.ylim(ymin, ymax)
#
#             save_path = os.path.join(output_directory, "loss.svg")
#             plt.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")
#             plt.close()
#             logger.info(f"Saved loss curve to {save_path}")
#         else:
#             logger.warning("No epoch loss history found on model; skip loss plot.")
#     except Exception as e:
#         logger.error(f"Failed to plot loss curve: {e}")
#     # ===========================================================
#
#     ckpt_path = trainer.checkpoint_callback.best_model_path
#     logger.info(f"Reloading best model from checkpoint file: {ckpt_path}")
#     fastprop_model = fastprop_model.__class__.load_from_checkpoint(ckpt_path)
#     validation_results = trainer.validate(fastprop_model, val_dataloader, verbose=False)
#     test_results = trainer.test(fastprop_model, test_dataloader, verbose=False)
#     validation_results_df = pd.DataFrame.from_records(validation_results, index=("value",))
#     logger.info("Displaying validation results for repetition %d:\n%s", repetition_number, validation_results_df.transpose().to_string())
#     test_results_df = pd.DataFrame.from_records(test_results, index=("value",))
#     logger.info("Displaying validation results for repetition %d:\n%s", repetition_number, test_results_df.transpose().to_string())
#     return test_results, validation_results
