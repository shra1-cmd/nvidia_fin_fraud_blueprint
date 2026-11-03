# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.

import os
import sys
import json
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb

from torch_geometric.nn import (
    HeteroConv,
    GATConv,
    TransformerConv,
    GeneralConv,
    SAGEConv,
)

from captum.attr import ShapleyValueSampling

# Triton Python backend utilities.
import triton_python_backend_utils as pb_utils

from json_loader_writer import save_meta, load_meta
from pathlib import Path
from typing import Dict, Union, Tuple, Optional, List

TensorKey = Union[str, Tuple[str, str, str]]


def setup_logging(level=logging.DEBUG):
    """Configure logging format and level."""
    logging.basicConfig(
        level=level,
        format=("%(asctime)s %(levelname)-8s " "%(filename)s:%(lineno)d — %(message)s"),
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_reverse_edges_new(
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
):
    """Add reverse edges for heterogeneous graphs (node prediction doesn't use edge attrs)."""
    new_edge_index = dict(edge_index_dict)

    for (src, rel, dst), ei in edge_index_dict.items():
        rev_key: Tuple[str, str, str] = (dst, f"rev_{rel}", src)
        new_edge_index[rev_key] = ei.flip(0)  # swap rows

    return new_edge_index


class HeterogeneousNodePredictor(nn.Module):
    """
    A heterogeneous GNN model for node classification.

    This model is adapted from HeterogeneousLinkPredictor but modified for node prediction.
    It applies message passing across the heterogeneous graph and produces node-level predictions.
    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        meta: dict,
        node_type_to_predict: str,
        conv_class: str = "GATConv",
        num_layers: int = 2,
        dropout_prob: float = 0.2,
    ):
        """
        Initialize the heterogeneous node predictor.

        Parameters:
        ----------
        hidden_channels : int
            Number of hidden channels in the GNN layers
        out_channels : int
            Number of output classes
        meta : dict
            Metadata dictionary containing node and edge type information
        node_type_to_predict : str
            The node type for which we're making predictions
        conv_class : str
            Type of GNN layer ("GATConv", "SAGEConv", "TransformerConv", "GeneralConv")
        num_layers : int
            Number of message passing layers
        dropout_prob : float
            Dropout probability
        """
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.node_type_to_predict = node_type_to_predict
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        class_name_to_conv_layer_dict = {
            "GeneralConv": GeneralConv,
            "TransformerConv": TransformerConv,
            "GATConv": GATConv,
            "SAGEConv": SAGEConv,
        }

        if conv_class not in class_name_to_conv_layer_dict:
            raise ValueError(
                f"Unknown conv_class: {conv_class}. "
                f"Available options: {list(class_name_to_conv_layer_dict.keys())}"
            )

        ConvLayer = class_name_to_conv_layer_dict[conv_class]

        # Input projection layers for each node type
        self.input_proj = nn.ModuleDict()
        for node_type, node_info in meta["nodes"].items():
            feat_dim = node_info["feat_dim"]
            self.input_proj[node_type] = nn.Linear(feat_dim, hidden_channels)

        # Message passing layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in meta["edges"].keys():
                # Only GATConv supports add_self_loops parameter
                # For heterogeneous graphs, disable self-loops when supported
                if conv_class == "GATConv":
                    conv_dict[edge_type] = ConvLayer(
                        hidden_channels, hidden_channels, add_self_loops=False
                    )
                else:
                    # All other conv types don't have add_self_loops parameter
                    conv_dict[edge_type] = ConvLayer(hidden_channels, hidden_channels)
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # Output layer (node classifier)
        # Concatenate original features + learned embeddings
        # Input dimension: original_feat_dim + hidden_channels
        original_feat_dim = meta["nodes"][node_type_to_predict]["feat_dim"]
        classifier_input_dim = original_feat_dim + hidden_channels

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        node_type: str,
        return_hidden: bool = False,
    ):
        """
        Forward pass through the model.

        Parameters:
        ----------
        x_dict : Dict[str, torch.Tensor]
            Dictionary of node features for each node type
        edge_index_dict : Dict[Tuple[str, str, str], torch.Tensor]
            Dictionary of edge indices for each edge type
        node_type : str
            The node type for which to make predictions
        return_hidden : bool
            If True, return the concatenated features (original + embeddings) instead of logits

        Returns:
        -------
        torch.Tensor
            Either logits (if return_hidden=False) or concatenated features (if return_hidden=True)
        """
        # Store original features for concatenation
        x_dict_original = x_dict

        # Input projection
        h_dict = {}
        for ntype, x in x_dict.items():
            h_dict[ntype] = self.input_proj[ntype](x)

        # Message passing
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {key: F.relu(h) for key, h in h_dict.items()}
            h_dict = {
                key: F.dropout(h, p=self.dropout_prob, training=self.training)
                for key, h in h_dict.items()
            }

        # Get embeddings for the target node type
        node_embeddings = h_dict[node_type]

        # Get original features for the target node type
        node_original_features = x_dict_original[node_type]

        # Concatenate original features with learned embeddings
        # Similar to HeterogeneousLinkPredictor's approach
        node_combined = torch.cat([node_original_features, node_embeddings], dim=-1)

        if return_hidden:
            return node_combined
        else:
            # Classify using both original features and learned embeddings
            return self.classifier(node_combined)


class TritonPythonModel:
    def initialize(self, args):

        model_config = json.loads(args["model_config"])
        parameters = model_config["parameters"]

        self.hidden_channels = int(parameters["hidden_channels"]["string_value"])
        self.out_channels = int(parameters["out_channels"]["string_value"])

        self.embedder_state_dict_filename = parameters[
            "embedding_generator_model_state_dict"
        ]["string_value"]

        self.xgb_model_filename = parameters["embeddings_based_xgboost_model"][
            "string_value"
        ]

        # For node prediction, we get the node type to predict
        self.node_type_to_predict = parameters["node_type_to_predict"]["string_value"]

        # GNN hyperparameters
        conv_class = parameters["conv_class"]["string_value"]
        num_layers = int(parameters["num_layers"]["string_value"])

        self.device = torch.device("cuda")

        current_directory = os.path.dirname(os.path.abspath(__file__))
        json_path = Path(current_directory) / "meta.json"
        meta = load_meta(json_path)

        self.model = HeterogeneousNodePredictor(
            self.hidden_channels,
            self.out_channels,
            meta=meta,
            node_type_to_predict=self.node_type_to_predict,
            conv_class=conv_class,
            num_layers=num_layers,
        ).to(self.device)

        self.meta = meta

        current_directory = os.path.dirname(os.path.abspath(__file__))
        state_dict = torch.load(
            os.path.join(current_directory, self.embedder_state_dict_filename),
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.bst = xgb.Booster()
        self.bst.load_model(os.path.join(current_directory, self.xgb_model_filename))
        self.bst.set_param({"tree_method": "hist", "device": "cuda"})

        # Precompute input schema info
        self.node_dims = {}
        self.edge_index_names = []
        for inp in model_config.get("input", []):
            name = inp["name"]
            dims = inp.get("dims", [])
            if name.startswith("x_"):
                node = name[len("x_") :]
                self.node_dims[node] = dims[1]
            elif name.startswith("edge_index_"):
                self.edge_index_names.append(name)

        # Determine if compute_shap input exists
        self.has_shap_flag = any(
            inp["name"] == "COMPUTE_SHAP" for inp in model_config.get("input", [])
        )

        self.device = torch.device("cuda")
        prediction_config = pb_utils.get_output_config_by_name(
            model_config, "PREDICTION"
        )

        self.prediction_dtype = pb_utils.triton_string_to_numpy(
            prediction_config["data_type"]
        )

    def execute(self, requests):
        responses = []
        x_dict = {}
        feature_mask = {}
        edge_index = {}

        def _forward_wrapped(
            input_keys: List[TensorKey], static: Dict[TensorKey, torch.Tensor], *inputs
        ):
            """Wrapper for model forward pass for SHAP computation."""
            data = {**static}
            x_dict_modified = {}

            for k, x in zip(input_keys, inputs):
                data[k] = x
                if k in x_dict:
                    x_dict_modified[k] = x

            edge_index_ = add_reverse_edges_new(edge_index)

            embeddings = self.model(
                x_dict_modified,
                edge_index_,
                node_type=self.node_type_to_predict,
                return_hidden=True,
            )

            out = self.bst.predict(xgb.DMatrix(embeddings.detach()))[:, None]
            out = out[0]

            return (
                out
                if isinstance(out, torch.Tensor)
                else torch.tensor(out, dtype=torch.float32, device=embeddings.device)
            )

        def _prepare_inputs_and_masks_generic(
            data: Dict[TensorKey, torch.Tensor],
            group_masks: Dict[TensorKey, torch.Tensor],
            static_keys: Optional[List[TensorKey]] = None,
            baseline_mode: str = "zeros",
        ) -> Tuple[
            Tuple[torch.Tensor, ...],
            Tuple[torch.Tensor, ...],
            Tuple[Optional[torch.Tensor], ...],
            List[TensorKey],
        ]:
            """Prepare inputs and masks for SHAP computation."""
            inputs: List[torch.Tensor] = []
            baselines: List[torch.Tensor] = []
            masks: List[Optional[torch.Tensor]] = []
            input_keys: List[TensorKey] = []

            static_set = set(static_keys or [])

            for key, x_key in data.items():
                # Skip anything designated static (e.g., edge_index)
                if key in static_set:
                    continue

                # We only treat 2D tensors as feature matrices (nodes)
                if not (isinstance(x_key, torch.Tensor) and x_key.dim() == 2):
                    continue

                _, num_cols = x_key.shape
                device = x_key.device

                # Baseline
                if baseline_mode == "zeros":
                    baseline = torch.zeros_like(x_key)
                elif baseline_mode == "mean":
                    baseline = x_key.mean(dim=0, keepdim=True).expand_as(x_key)
                else:
                    raise ValueError("Unsupported baseline_mode")

                # Group mask
                gids = group_masks.get(key, None)
                if gids is None:
                    # default: each column is its own group
                    gids = torch.arange(num_cols, device=device, dtype=torch.long)
                else:
                    gids = gids.to(device=device, dtype=torch.long).view(-1)
                    assert (
                        gids.numel() == num_cols
                    ), f"Mask for key {key} must have length {num_cols} (got {gids.numel()})"

                # IMPORTANT for Shapley with scalar outputs: use [1, num_cols]
                feature_mask = gids.unsqueeze(0)  # [1, num_cols]

                inputs.append(x_key)
                baselines.append(baseline)
                masks.append(feature_mask)
                input_keys.append(key)

            return tuple(inputs), tuple(baselines), tuple(masks), input_keys

        def _attribute_per_raw_feature(
            data: Dict[TensorKey, torch.Tensor],
            group_masks: Dict[TensorKey, torch.Tensor],
            static_keys: Optional[List[TensorKey]] = None,
            method: str = "shapley",
            n_samples: int = 128,
            reduction: str = "sum_abs",
        ):
            """Compute SHAP values for node features."""
            # Separate static tensors (kept fixed) from variable inputs (nodes only)
            static = {k: data[k] for k in (static_keys or []) if k in data}

            inputs, baselines, masks, input_keys = _prepare_inputs_and_masks_generic(
                data=data, group_masks=group_masks, static_keys=static_keys
            )

            # Build explainer & compute attributions
            if method == "shapley":
                explainer = ShapleyValueSampling(
                    lambda *inp: _forward_wrapped(input_keys, static, *inp)
                )
                attr_list = explainer.attribute(
                    inputs=inputs,
                    baselines=baselines,
                    feature_mask=masks,
                    n_samples=n_samples,
                )
            else:
                raise ValueError("method must be 'shapley'")

            # Aggregate: per-column -> per-raw-feature group using group_masks[key]
            results: Dict[str, torch.Tensor] = {}
            for key, attributions in zip(input_keys, attr_list):
                # attributions: [N, F] attributions (same shape as the input matrix)
                col_imp = attributions.abs().mean(dim=0)  # [F], mean across rows

                # Get the grouping ids for this key
                gids = group_masks.get(key, None)
                if gids is None:
                    gids = torch.arange(
                        col_imp.numel(), device=attributions.device, dtype=torch.long
                    )
                else:
                    gids = gids.to(device=attributions.device, dtype=torch.long).view(
                        -1
                    )

                valid = gids >= 0
                gids_valid = gids[valid]
                col_imp_valid = col_imp[valid]

                unique_ids, _ = torch.sort(torch.unique(gids_valid))
                group_scores = []
                for gid in unique_ids.tolist():
                    m = gids_valid == gid
                    score = (
                        col_imp_valid[m].sum()
                        if reduction == "sum_abs"
                        else col_imp_valid[m].mean()
                    )
                    group_scores.append(score)

                # Build result keys (only node keys for node prediction)
                if isinstance(key, str):
                    # node key
                    out_prefix = f"{key}"
                    results[f"shap_values_{out_prefix}"] = (
                        torch.stack(group_scores).detach().cpu()
                    )
                    results[f"shap_groups_{out_prefix}"] = unique_ids.detach().cpu()

            return results

        for request in requests:
            # Compute SHAP flag
            compute_shap = False
            if self.has_shap_flag:
                comp = pb_utils.get_input_tensor_by_name(
                    request, "COMPUTE_SHAP"
                ).as_numpy()
                compute_shap = bool(comp.item())

            # Parse node features and masks
            for node, feat_dim in self.node_dims.items():
                arr = pb_utils.get_input_tensor_by_name(request, f"x_{node}").as_numpy()
                x_dict[node] = torch.from_numpy(arr).to(self.device)
                if compute_shap:
                    mask = pb_utils.get_input_tensor_by_name(
                        request, f"feature_mask_{node}"
                    ).as_numpy()
                    feature_mask[node] = torch.from_numpy(mask).int().to(self.device)

            # Parse edge indices (no edge attributes for node prediction)
            for name in self.edge_index_names:
                arr = pb_utils.get_input_tensor_by_name(request, name).as_numpy()
                edge = tuple(name[len("edge_index_") :].split("_"))
                edge_index[edge] = torch.from_numpy(arr).long().to(self.device)

            # Run model
            edge_index_ = add_reverse_edges_new(edge_index)
            embeddings = self.model(
                x_dict,
                edge_index_,
                node_type=self.node_type_to_predict,
                return_hidden=True,
            )

            y_pred_prob = self.bst.predict(xgb.DMatrix(embeddings.detach()))[:, None]

            output_tensors = [
                pb_utils.Tensor(
                    "PREDICTION",
                    y_pred_prob.astype(self.prediction_dtype),
                )
            ]

            if compute_shap:
                data = {}
                static_keys = []
                for key in x_dict.keys():
                    data[key] = x_dict[key]

                results = _attribute_per_raw_feature(
                    data=data,
                    group_masks=feature_mask,
                    static_keys=static_keys,
                    method="shapley",
                    n_samples=64,
                    reduction="sum_abs",
                )

                for node in self.node_dims:
                    shap_val = results.get(f"shap_values_{node}")
                    if shap_val is not None:
                        shap_np = shap_val.detach().cpu().numpy()
                        output_tensors.append(
                            pb_utils.Tensor(f"shap_values_{node}", shap_np)
                        )
            else:
                # Return zero SHAP values when not computing
                for node_name, node_info in self.meta["nodes"].items():
                    feat_dim = node_info["feat_dim"]
                    shap_numpy_array = np.zeros(feat_dim, dtype=np.float32)
                    output_tensors.append(
                        pb_utils.Tensor(f"shap_values_{node_name}", shap_numpy_array)
                    )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=output_tensors
            )

            responses.append(inference_response)

        return responses
