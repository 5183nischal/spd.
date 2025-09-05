#!/usr/bin/env python3
"""
Standalone Research Implementation of Stochastic Parameter Decomposition (SPD) on TMS

This script implements SPD (Stochastic Parameter Decomposition) as described in the paper:
"Stochastic Parameter Decomposition" by Bushnaq, Braun, and Sharkey.

SPD decomposes neural network parameters into rank-one subcomponents and learns causal 
importance functions to predict which components can be ablated for each input. This is
more scalable and robust than Attribution-based Parameter Decomposition (APD).

Mathematical Framework:
- Parameter decomposition: W^l_{i,j} ≈ Σ_c U^l_{i,c} V^l_{c,j}
- Causal importance: g^l_c(x) = σ_H(γ^l_c(h^l_c(x)))
- Stochastic masking: m^l_c(x,r) = g^l_c(x) + (1-g^l_c(x))r^l_c where r~U(0,1)
- Combined loss: L_SPD = L_faithfulness + β₁L_stoch_recon + β₂L_stoch_recon_layer + β₃L_importance_min

Author: Research implementation based on original SPD codebase
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

@dataclass
class TMSConfig:
    """Configuration for Toy Model of Superposition"""
    n_features: int = 5          # Number of input features (m2)
    n_hidden: int = 2            # Hidden dimension - bottleneck (m1)
    feature_probability: float = 0.05  # Sparsity of input features
    tied_weights: bool = False   # Whether to tie weights (W2 = W1^T)
    bias: bool = True           # Whether to use bias in output layer
    
@dataclass 
class SPDConfig:
    """Configuration for SPD decomposition"""
    C: int = 20                 # Number of components per layer
    n_mask_samples: int = 1     # Number of stochastic mask samples per step
    gate_hidden_dims: List[int] = None  # Hidden dims for causal importance MLPs
    
    # Loss coefficients (from paper and configs)
    faithfulness_coeff: float = 1.0
    stochastic_recon_coeff: float = 1.0
    stochastic_recon_layerwise_coeff: float = 1.0  
    importance_minimality_coeff: float = 3e-3
    pnorm: float = 1.0          # p-norm for importance minimality loss
    
    def __post_init__(self):
        if self.gate_hidden_dims is None:
            self.gate_hidden_dims = [16]

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 4096
    steps: int = 5000           # Reduced for demo - paper uses 40k
    lr: float = 1e-3
    lr_schedule: Literal["cosine", "linear", "constant"] = "cosine"
    lr_warmup_pct: float = 0.0
    eval_freq: int = 500
    print_freq: int = 100
    
    # For TMS training
    tms_steps: int = 20000
    tms_lr: float = 1e-3
    tms_importance: float = 1.0  # Feature importance weighting


class SparseFeatureDataset(Dataset):
    """
    Dataset for sparse feature generation as used in TMS.
    Generates sparse vectors where each feature activates independently 
    with probability feature_probability.
    """
    
    def __init__(self, n_features: int, feature_probability: float, 
                 batch_size: int, device: str = "cpu"):
        self.n_features = n_features
        self.feature_probability = feature_probability
        self.batch_size = batch_size
        self.device = device
        
    def __len__(self):
        return 100000  # Large number for sampling
        
    def __getitem__(self, idx):
        # Generate batch of sparse features
        batch = torch.rand(self.batch_size, self.n_features, device=self.device)
        mask = torch.rand_like(batch) < self.feature_probability
        batch = batch * mask.float()
        
        # Labels are the same as inputs for reconstruction task
        return batch, batch


def infinite_dataloader(dataloader):
    """Create an infinite iterator from a dataloader"""
    while True:
        for batch in dataloader:
            yield batch
        

class TMSModel(nn.Module):
    """
    Toy Model of Superposition as described in Elhage et al. 2022.
    
    Architecture: x̂ = ReLU(W2 * W1 * x + b)
    Where W1 ∈ R^(n_hidden × n_features), W2 ∈ R^(n_features × n_hidden)
    
    The model learns to represent features in superposition when n_hidden < n_features.
    """
    
    def __init__(self, config: TMSConfig):
        super().__init__()
        self.config = config
        
        # First layer: features -> hidden (compression)
        self.linear1 = nn.Linear(config.n_features, config.n_hidden, bias=False)
        
        # Second layer: hidden -> features (decompression)  
        self.linear2 = nn.Linear(config.n_hidden, config.n_features, bias=config.bias)
        
        if config.tied_weights:
            self.tie_weights()
            
    def tie_weights(self):
        """Tie weights so that W2 = W1^T"""
        self.linear2.weight.data = self.linear1.weight.data.T
        
    def forward(self, x):
        hidden = self.linear1(x)
        out_pre_relu = self.linear2(hidden)
        out = F.relu(out_pre_relu)
        return out


class HardSigmoid(nn.Module):
    """Hard sigmoid activation function: σ_H(x) = clip((x + 1) / 2, 0, 1)"""
    
    def forward(self, x):
        return torch.clamp((x + 1) / 2, 0, 1)


class CausalImportanceMLP(nn.Module):
    """
    MLP that predicts causal importance g^l_c(x) for a single component.
    
    Takes scalar input h^l_c(x) = Σ_j V^l_{c,j} a^l_j(x) and outputs scalar in [0,1].
    Architecture: Linear -> GELU -> ... -> Linear -> HardSigmoid
    """
    
    def __init__(self, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        input_dim = 1  # Scalar input
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.GELU()
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, 1))  # Output scalar
        layers.append(HardSigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, h):
        """h: (..., ) scalar activations -> (..., ) causal importances"""
        h_expanded = h.unsqueeze(-1)  # (..., 1)
        output = self.network(h_expanded)  # (..., 1) 
        return output.squeeze(-1)  # (..., )


class LinearComponents(nn.Module):
    """
    Rank-one decomposition of a linear layer into C components.
    
    Each component c is parameterized by vectors U^l_c and V^l_c such that:
    W^l ≈ Σ_c U^l_c ⊗ V^l_c^T
    
    Also includes causal importance functions Γ^l_c for each component.
    """
    
    def __init__(self, input_dim: int, output_dim: int, C: int, 
                 gate_hidden_dims: List[int], device: str = "cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.C = C
        self.device = device
        
        # Rank-one components: U^l_c (output_dim,) and V^l_c (input_dim,)
        self.U = nn.Parameter(torch.randn(C, output_dim, device=device))
        self.V = nn.Parameter(torch.randn(C, input_dim, device=device))
        
        # Causal importance MLPs Γ^l_c
        self.causal_importance_mlps = nn.ModuleList([
            CausalImportanceMLP(gate_hidden_dims) for _ in range(C)
        ])
        
        # Initialize parameters
        with torch.no_grad():
            std = 1.0 / math.sqrt(input_dim)
            self.U.data.normal_(0, std)
            self.V.data.normal_(0, std)
    
    def compute_inner_activations(self, activations):
        """
        Compute inner activations h^l_c(x) = Σ_j V^l_{c,j} a^l_j(x)
        
        Args:
            activations: (batch, input_dim) - activations that get multiplied by weight matrix
        Returns:
            (batch, C) - inner activations for each component
        """
        # h^l_c(x) = V^l_c · a^l(x)
        return torch.einsum('bi,ci->bc', activations, self.V)
    
    def compute_causal_importances(self, activations):
        """
        Compute causal importances g^l_c(x) = Γ^l_c(h^l_c(x))
        
        Args:
            activations: (batch, input_dim)
        Returns:
            (batch, C) - causal importances in [0,1]
        """
        inner_activations = self.compute_inner_activations(activations)  # (batch, C)
        
        causal_importances = []
        for c in range(self.C):
            h_c = inner_activations[:, c]  # (batch,)
            g_c = self.causal_importance_mlps[c](h_c)  # (batch,)
            causal_importances.append(g_c)
            
        return torch.stack(causal_importances, dim=1)  # (batch, C)
    
    def compute_stochastic_masks(self, causal_importances, n_samples=1):
        """
        Compute stochastic masks: m^l_c(x,r) = g^l_c(x) + (1-g^l_c(x)) * r^l_c
        where r^l_c ~ U(0,1)
        
        Args:
            causal_importances: (batch, C)
            n_samples: number of random mask samples
        Returns:
            List of (batch, C) mask tensors
        """
        masks = []
        for _ in range(n_samples):
            r = torch.rand_like(causal_importances)  # (batch, C)
            mask = causal_importances + (1 - causal_importances) * r
            masks.append(mask)
        return masks
    
    def reconstruct_weight_matrix(self):
        """Reconstruct weight matrix: W = Σ_c U_c ⊗ V_c^T"""
        # W^l = Σ_c U^l_c V^l_c^T  
        return torch.einsum('co,ci->oi', self.U, self.V)
    
    def forward_with_masks(self, x, masks):
        """
        Forward pass with stochastic masks applied to components.
        
        Args:
            x: (batch, input_dim)
            masks: (batch, C) - masks to apply to each component
        Returns:
            (batch, output_dim) - output with masked components
        """
        batch_size = x.size(0)
        
        # Compute contribution of each masked component
        # For component c: mask[c] * U_c * (V_c^T @ x)
        
        # V_c^T @ x for all components: (C, batch, 1)
        v_activations = torch.einsum('ci,bi->cb', self.V, x)  # (C, batch)
        
        # Apply masks: (batch, C) 
        masked_activations = masks * v_activations.T  # (batch, C)
        
        # U_c * masked_activations: (batch, output_dim)
        output = torch.einsum('co,bc->bo', self.U, masked_activations)
        
        return output
    
    def forward(self, x):
        """Standard forward pass (all components active)"""
        ones_mask = torch.ones(x.size(0), self.C, device=x.device)
        return self.forward_with_masks(x, ones_mask)


class SPDComponentModel(nn.Module):
    """
    ComponentModel that wraps a target model and replaces specified layers 
    with LinearComponents for SPD decomposition.
    """
    
    def __init__(self, target_model: TMSModel, spd_config: SPDConfig, device: str = "cpu"):
        super().__init__()
        self.target_model = target_model
        self.spd_config = spd_config
        self.device = device
        
        # Create component versions of target layers
        self.components = nn.ModuleDict()
        
        # Replace linear1 
        self.components['linear1'] = LinearComponents(
            input_dim=target_model.linear1.in_features,
            output_dim=target_model.linear1.out_features, 
            C=spd_config.C,
            gate_hidden_dims=spd_config.gate_hidden_dims,
            device=device
        )
        
        # Replace linear2
        self.components['linear2'] = LinearComponents(
            input_dim=target_model.linear2.in_features,
            output_dim=target_model.linear2.out_features,
            C=spd_config.C, 
            gate_hidden_dims=spd_config.gate_hidden_dims,
            device=device
        )
        
        # Store bias separately (not decomposed)
        if target_model.linear2.bias is not None:
            self.bias = nn.Parameter(target_model.linear2.bias.clone())
        else:
            self.bias = None
    
    def forward_with_target_activations(self, x):
        """Forward pass storing intermediate activations from target model"""
        with torch.no_grad():
            target_hidden = self.target_model.linear1(x)
            target_out = self.target_model.linear2(target_hidden)
            if self.target_model.linear2.bias is not None:
                target_out = target_out + self.target_model.linear2.bias
            target_final = F.relu(target_out)
            
        return {
            'input': x,
            'hidden': target_hidden, 
            'pre_relu': target_out,
            'output': target_final
        }
    
    def compute_all_causal_importances(self, x):
        """Compute causal importances for all components"""
        causal_importances = {}
        
        # For linear1: input activations 
        causal_importances['linear1'] = self.components['linear1'].compute_causal_importances(x)
        
        # For linear2: need hidden activations from linear1
        # Use target model's activations for consistency
        with torch.no_grad():
            hidden = self.target_model.linear1(x)
        causal_importances['linear2'] = self.components['linear2'].compute_causal_importances(hidden)
        
        return causal_importances
    
    def forward_with_stochastic_masks(self, x, masks_dict):
        """
        Forward pass with stochastic masks applied to components.
        
        Args:
            x: input tensor
            masks_dict: Dict[str, Tensor] - masks for each layer
        """
        # Forward through masked linear1
        hidden = self.components['linear1'].forward_with_masks(x, masks_dict['linear1'])
        
        # Forward through masked linear2
        out = self.components['linear2'].forward_with_masks(hidden, masks_dict['linear2'])
        
        # Add bias if present
        if self.bias is not None:
            out = out + self.bias
            
        # Apply ReLU
        final_out = F.relu(out)
        
        return final_out
    
    def forward_layerwise_masked(self, x, layer_name, masks):
        """Forward pass with only one layer masked, others using target weights"""
        if layer_name == 'linear1':
            # Mask only linear1, use target linear2
            hidden = self.components['linear1'].forward_with_masks(x, masks)
            out = self.target_model.linear2(hidden)
            
        elif layer_name == 'linear2':
            # Use target linear1, mask only linear2  
            hidden = self.target_model.linear1(x)
            out = self.components['linear2'].forward_with_masks(hidden, masks)
            
        else:
            raise ValueError(f"Unknown layer: {layer_name}")
            
        return F.relu(out)
    
    def get_faithfulness_loss(self):
        """Compute faithfulness loss: ||W_target - W_reconstructed||^2"""
        loss = 0.0
        total_params = 0
        
        for layer_name, component in self.components.items():
            target_layer = getattr(self.target_model, layer_name)
            target_weight = target_layer.weight  # (out, in)
            reconstructed_weight = component.reconstruct_weight_matrix()  # (out, in)
            
            layer_loss = F.mse_loss(reconstructed_weight, target_weight, reduction='sum')
            loss += layer_loss
            total_params += target_weight.numel()
            
        return loss / total_params


def compute_stochastic_masks_all_layers(causal_importances_dict, n_samples=1):
    """Compute stochastic masks for all layers"""
    all_masks = []
    
    for _ in range(n_samples):
        sample_masks = {}
        for layer_name, ci in causal_importances_dict.items():
            r = torch.rand_like(ci)
            mask = ci + (1 - ci) * r
            sample_masks[layer_name] = mask
        all_masks.append(sample_masks)
        
    return all_masks


def compute_spd_losses(component_model, batch, target_activations, spd_config):
    """
    Compute all SPD loss terms:
    1. Faithfulness loss
    2. Stochastic reconstruction loss 
    3. Stochastic reconstruction layerwise loss
    4. Importance minimality loss
    """
    x, labels = batch
    target_output = target_activations['output']
    
    # 1. Faithfulness loss
    faithfulness_loss = component_model.get_faithfulness_loss()
    
    # 2. Compute causal importances
    causal_importances = component_model.compute_all_causal_importances(x)
    
    # 3. Stochastic reconstruction loss
    stochastic_masks = compute_stochastic_masks_all_layers(
        causal_importances, spd_config.n_mask_samples
    )
    
    stoch_recon_loss = 0.0
    for masks in stochastic_masks:
        masked_output = component_model.forward_with_stochastic_masks(x, masks)
        stoch_recon_loss += F.mse_loss(masked_output, target_output)
    stoch_recon_loss /= len(stochastic_masks)
    
    # 4. Stochastic reconstruction layerwise loss
    stoch_recon_layerwise_loss = 0.0
    layer_count = 0
    
    for layer_name in component_model.components.keys():
        for masks in stochastic_masks:
            layer_masks = masks[layer_name]
            layerwise_output = component_model.forward_layerwise_masked(x, layer_name, layer_masks)
            stoch_recon_layerwise_loss += F.mse_loss(layerwise_output, target_output)
            layer_count += 1
    
    if layer_count > 0:
        stoch_recon_layerwise_loss /= layer_count
    
    # 5. Importance minimality loss
    importance_min_loss = 0.0
    for layer_name, ci in causal_importances.items():
        # ||g^l_c(x)||_p^p
        importance_min_loss += torch.sum(torch.abs(ci) ** spd_config.pnorm)
    
    # Average over batch
    batch_size = x.size(0)
    importance_min_loss /= batch_size
    
    # Combine losses
    total_loss = (
        spd_config.faithfulness_coeff * faithfulness_loss +
        spd_config.stochastic_recon_coeff * stoch_recon_loss +  
        spd_config.stochastic_recon_layerwise_coeff * stoch_recon_layerwise_loss +
        spd_config.importance_minimality_coeff * importance_min_loss
    )
    
    loss_dict = {
        'total': total_loss.item(),
        'faithfulness': faithfulness_loss.item(),
        'stoch_recon': stoch_recon_loss.item(),
        'stoch_recon_layerwise': stoch_recon_layerwise_loss.item(),
        'importance_min': importance_min_loss.item()
    }
    
    return total_loss, loss_dict


def compute_ci_l0(causal_importances_dict, threshold=0.1):
    """Compute L0 norm of causal importances (number of active components)"""
    total_active = 0
    total_components = 0
    
    for layer_name, ci in causal_importances_dict.items():
        active = (ci > threshold).float().sum(-1).mean()  # Average over batch
        total_active += active.item()
        total_components += ci.size(-1)
        
    return total_active, total_components


def get_lr_schedule_fn(schedule_type: str):
    """Get learning rate schedule function"""
    if schedule_type == "cosine":
        return lambda step, total_steps: 0.5 * (1 + math.cos(math.pi * step / total_steps))
    elif schedule_type == "linear":
        return lambda step, total_steps: 1 - step / total_steps
    elif schedule_type == "constant":
        return lambda step, total_steps: 1.0
    else:
        raise ValueError(f"Unknown lr schedule: {schedule_type}")


def train_tms_model(config: TMSConfig, train_config: TrainingConfig, device: str):
    """Train a TMS model on sparse features (this becomes our target model)"""
    print("=== Training TMS Target Model ===")
    
    # Create model and dataset
    model = TMSModel(config).to(device)
    dataset = SparseFeatureDataset(
        n_features=config.n_features,
        feature_probability=config.feature_probability,
        batch_size=train_config.batch_size,
        device=device
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.tms_lr)
    lr_schedule_fn = get_lr_schedule_fn(train_config.lr_schedule)
    
    # Training loop
    model.train()
    data_iter = infinite_dataloader(dataloader)
    
    for step in tqdm(range(train_config.tms_steps), desc="Training TMS"):
        # Get batch
        batch, labels = next(data_iter)
        batch, labels = batch.squeeze(0), labels.squeeze(0)  # Remove dataloader batch dim
        
        # Forward pass
        output = model(batch)
        
        # Loss: importance-weighted reconstruction (as in original TMS)
        error = train_config.tms_importance * (labels.abs() - output) ** 2
        loss = error.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        step_lr = train_config.tms_lr * lr_schedule_fn(step, train_config.tms_steps)
        for group in optimizer.param_groups:
            group['lr'] = step_lr
            
        # Logging
        if step % train_config.print_freq == 0:
            print(f"Step {step:5d}, Loss: {loss.item():.6f}, LR: {step_lr:.6f}")
            
    print(f"TMS training completed. Final loss: {loss.item():.6f}")
    
    # Tie weights if specified
    if config.tied_weights:
        model.tie_weights()
        
    return model


def train_spd_decomposition(target_model: TMSModel, tms_config: TMSConfig, 
                          spd_config: SPDConfig, train_config: TrainingConfig, device: str):
    """Train SPD decomposition of the target TMS model"""
    print("\n=== Training SPD Decomposition ===")
    
    # Create component model
    component_model = SPDComponentModel(target_model, spd_config, device)
    
    # Create dataset (same as used for TMS training)
    dataset = SparseFeatureDataset(
        n_features=tms_config.n_features,
        feature_probability=tms_config.feature_probability, 
        batch_size=train_config.batch_size,
        device=device
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(component_model.parameters(), lr=train_config.lr)
    lr_schedule_fn = get_lr_schedule_fn(train_config.lr_schedule)
    
    # Training loop
    target_model.eval()  # Target model stays in eval mode
    component_model.train()
    data_iter = infinite_dataloader(dataloader)
    
    for step in tqdm(range(train_config.steps), desc="Training SPD"):
        # Get batch
        batch, _ = next(data_iter)
        batch = batch.squeeze(0)  # Remove dataloader batch dim
        x = batch  # Use the whole batch as input and labels
        
        # Get target activations
        target_activations = component_model.forward_with_target_activations(x)
        
        # Compute SPD losses
        total_loss, loss_dict = compute_spd_losses(
            component_model, (x, x), target_activations, spd_config
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update learning rate
        step_lr = train_config.lr * lr_schedule_fn(step, train_config.steps)
        for group in optimizer.param_groups:
            group['lr'] = step_lr
        
        # Logging and evaluation
        if step % train_config.print_freq == 0:
            print(f"Step {step:5d}, Total Loss: {loss_dict['total']:.6f}")
            print(f"  Faithfulness: {loss_dict['faithfulness']:.6f}, " +
                  f"Stoch Recon: {loss_dict['stoch_recon']:.6f}")
            print(f"  Stoch Recon Layer: {loss_dict['stoch_recon_layerwise']:.6f}, " +
                  f"Importance Min: {loss_dict['importance_min']:.6f}")
            
        if step % train_config.eval_freq == 0:
            # Compute metrics
            with torch.no_grad():
                causal_importances = component_model.compute_all_causal_importances(x)
                active_components, total_components = compute_ci_l0(causal_importances)
                print(f"  Active components: {active_components:.1f}/{total_components}")
                
    print(f"SPD training completed. Final loss: {loss_dict['total']:.6f}")
    return component_model


def analyze_decomposition(target_model: TMSModel, component_model: SPDComponentModel, 
                         tms_config: TMSConfig, device: str):
    """Analyze the learned decomposition and compare with ground truth"""
    print("\n=== Analyzing Decomposition ===")
    
    target_model.eval()
    component_model.eval()
    
    with torch.no_grad():
        # Get target weights
        target_W1 = target_model.linear1.weight.data  # (n_hidden, n_features)
        target_W2 = target_model.linear2.weight.data  # (n_features, n_hidden)
        
        print(f"Target W1 shape: {target_W1.shape}")
        print(f"Target W2 shape: {target_W2.shape}")
        
        # Get reconstructed weights
        recon_W1 = component_model.components['linear1'].reconstruct_weight_matrix()
        recon_W2 = component_model.components['linear2'].reconstruct_weight_matrix()
        
        print(f"Reconstructed W1 shape: {recon_W1.shape}")
        print(f"Reconstructed W2 shape: {recon_W2.shape}")
        
        # Compute reconstruction errors
        w1_error = F.mse_loss(recon_W1, target_W1).item()
        w2_error = F.mse_loss(recon_W2, target_W2).item()
        
        print(f"W1 reconstruction MSE: {w1_error:.6f}")
        print(f"W2 reconstruction MSE: {w2_error:.6f}")
        
        # Sample some test data to analyze causal importances
        test_x = torch.rand(32, tms_config.n_features, device=device)
        test_mask = torch.rand_like(test_x) < tms_config.feature_probability
        test_x = test_x * test_mask.float()
        
        causal_importances = component_model.compute_all_causal_importances(test_x)
        
        for layer_name, ci in causal_importances.items():
            mean_ci = ci.mean(dim=0)  # Average over batch
            active_components = (mean_ci > 0.1).sum().item()
            print(f"{layer_name} - Active components (>0.1): {active_components}/{ci.size(1)}")
            print(f"  Mean CI range: [{mean_ci.min():.3f}, {mean_ci.max():.3f}]")
        
        # Analyze component U and V vectors for interpretability
        print("\n--- Component Analysis ---")
        for layer_name, component in component_model.components.items():
            U = component.U.data  # (C, output_dim)
            V = component.V.data  # (C, input_dim)
            
            # Compute component norms
            component_norms = torch.norm(U, dim=1) * torch.norm(V, dim=1)
            top_components = torch.topk(component_norms, k=min(10, len(component_norms)))
            
            print(f"{layer_name} - Top component norms: {top_components.values.cpu().numpy()}")
            print(f"  Component indices: {top_components.indices.cpu().numpy()}")


def visualize_results(target_model: TMSModel, component_model: SPDComponentModel, 
                     tms_config: TMSConfig, device: str):
    """Create visualizations of the decomposition results"""
    print("\n=== Creating Visualizations ===")
    
    with torch.no_grad():
        # Plot target vs reconstructed weights
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # W1 comparison
        target_W1 = target_model.linear1.weight.data.cpu().numpy()
        recon_W1 = component_model.components['linear1'].reconstruct_weight_matrix().cpu().numpy()
        
        im1 = axes[0, 0].imshow(target_W1, cmap='RdBu_r', aspect='auto')
        axes[0, 0].set_title('Target W1')
        axes[0, 0].set_xlabel('Input Features')
        axes[0, 0].set_ylabel('Hidden Units')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(recon_W1, cmap='RdBu_r', aspect='auto')
        axes[0, 1].set_title('Reconstructed W1')
        axes[0, 1].set_xlabel('Input Features')
        axes[0, 1].set_ylabel('Hidden Units')
        plt.colorbar(im2, ax=axes[0, 1])
        
        diff_W1 = target_W1 - recon_W1
        im3 = axes[0, 2].imshow(diff_W1, cmap='RdBu_r', aspect='auto')
        axes[0, 2].set_title('Difference W1')
        axes[0, 2].set_xlabel('Input Features')
        axes[0, 2].set_ylabel('Hidden Units')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # W2 comparison
        target_W2 = target_model.linear2.weight.data.cpu().numpy()
        recon_W2 = component_model.components['linear2'].reconstruct_weight_matrix().cpu().numpy()
        
        im4 = axes[1, 0].imshow(target_W2, cmap='RdBu_r', aspect='auto')
        axes[1, 0].set_title('Target W2')
        axes[1, 0].set_xlabel('Hidden Units')
        axes[1, 0].set_ylabel('Output Features')
        plt.colorbar(im4, ax=axes[1, 0])
        
        im5 = axes[1, 1].imshow(recon_W2, cmap='RdBu_r', aspect='auto')
        axes[1, 1].set_title('Reconstructed W2')
        axes[1, 1].set_xlabel('Hidden Units')
        axes[1, 1].set_ylabel('Output Features')
        plt.colorbar(im5, ax=axes[1, 1])
        
        diff_W2 = target_W2 - recon_W2
        im6 = axes[1, 2].imshow(diff_W2, cmap='RdBu_r', aspect='auto')
        axes[1, 2].set_title('Difference W2')
        axes[1, 2].set_xlabel('Hidden Units')
        axes[1, 2].set_ylabel('Output Features')
        plt.colorbar(im6, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('spd_weight_comparison.png', dpi=150, bbox_inches='tight')
        print("Saved weight comparison plot to 'spd_weight_comparison.png'")
        
        # Plot component importance distribution
        test_x = torch.rand(100, tms_config.n_features, device=device)
        test_mask = torch.rand_like(test_x) < tms_config.feature_probability
        test_x = test_x * test_mask.float()
        
        causal_importances = component_model.compute_all_causal_importances(test_x)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, (layer_name, ci) in enumerate(causal_importances.items()):
            ci_mean = ci.mean(dim=0).cpu().numpy()
            axes[i].bar(range(len(ci_mean)), ci_mean)
            axes[i].set_title(f'{layer_name} - Average Causal Importance')
            axes[i].set_xlabel('Component Index')
            axes[i].set_ylabel('Causal Importance')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('spd_causal_importance.png', dpi=150, bbox_inches='tight')
        print("Saved causal importance plot to 'spd_causal_importance.png'")


def main():
    """Main function to run the complete SPD experiment"""
    print("=" * 60)
    print("STANDALONE SPD (Stochastic Parameter Decomposition) on TMS")
    print("=" * 60)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Configurations
    tms_config = TMSConfig(
        n_features=5,
        n_hidden=2, 
        feature_probability=0.05,
        tied_weights=False,
        bias=True
    )
    
    spd_config = SPDConfig(
        C=20,
        n_mask_samples=1,
        gate_hidden_dims=[16],
        faithfulness_coeff=1.0,
        stochastic_recon_coeff=1.0,
        stochastic_recon_layerwise_coeff=1.0,
        importance_minimality_coeff=3e-3,
        pnorm=1.0
    )
    
    train_config = TrainingConfig(
        batch_size=4096,
        steps=1000,  # Reduced for testing
        lr=1e-3,
        lr_schedule="cosine",
        eval_freq=200,
        print_freq=50,
        tms_steps=2000,  # Reduced for testing
        tms_lr=1e-3
    )
    
    print("Configurations:")
    print(f"  TMS: {tms_config.n_features} features → {tms_config.n_hidden} hidden")
    print(f"  SPD: {spd_config.C} components per layer")
    print(f"  Training: {train_config.tms_steps} TMS steps, {train_config.steps} SPD steps")
    
    # Phase 1: Train target TMS model
    target_model = train_tms_model(tms_config, train_config, device)
    
    # Phase 2: Train SPD decomposition
    component_model = train_spd_decomposition(target_model, tms_config, spd_config, train_config, device)
    
    # Phase 3: Analysis and visualization
    analyze_decomposition(target_model, component_model, tms_config, device)
    visualize_results(target_model, component_model, tms_config, device)
    
    print("\n" + "=" * 60)
    print("SPD EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("Check the generated plots for visualization of results.")
    print("=" * 60)


if __name__ == "__main__":
    main()