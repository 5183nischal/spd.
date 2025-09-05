# SPD Usage Guide and Experimental Notes

## Quick Start

To run the standalone SPD implementation on TMS:

```bash
python standalone_spd_tms.py
```

This will:
1. Train a TMS target model (5 features → 2 hidden dimensions)
2. Apply SPD to decompose the trained model into rank-one components
3. Analyze the decomposition and generate visualizations
4. Save plots showing weight comparisons and causal importance distributions

## Customizing the Experiment

### Model Configuration

```python
tms_config = TMSConfig(
    n_features=5,           # Number of input features
    n_hidden=2,            # Hidden dimension (bottleneck)
    feature_probability=0.05,  # Sparsity of input features
    tied_weights=False,    # Whether W2 = W1^T
    bias=True             # Use bias in output layer
)
```

### SPD Configuration

```python
spd_config = SPDConfig(
    C=20,                 # Number of components per layer
    n_mask_samples=1,     # Stochastic mask samples per step
    gate_hidden_dims=[16], # Hidden dimensions for causal importance MLPs
    
    # Loss coefficients
    faithfulness_coeff=1.0,
    stochastic_recon_coeff=1.0,
    stochastic_recon_layerwise_coeff=1.0,
    importance_minimality_coeff=3e-3,
    pnorm=1.0            # p-norm for importance minimality
)
```

### Training Configuration

```python
train_config = TrainingConfig(
    batch_size=4096,
    steps=5000,           # SPD training steps
    lr=1e-3,
    lr_schedule="cosine",
    
    # TMS training
    tms_steps=20000,
    tms_lr=1e-3,
    tms_importance=1.0
)
```

## Experiment Variations

### 1. Different TMS Sizes

```python
# Smaller model
tms_config = TMSConfig(n_features=3, n_hidden=2)

# Larger model  
tms_config = TMSConfig(n_features=40, n_hidden=10)

# More bottleneck pressure
tms_config = TMSConfig(n_features=10, n_hidden=3)
```

### 2. Sparsity Levels

```python
# Very sparse (harder)
tms_config = TMSConfig(feature_probability=0.01)

# Less sparse (easier)
tms_config = TMSConfig(feature_probability=0.1)

# Dense (no superposition)
tms_config = TMSConfig(feature_probability=0.5)
```

### 3. Component Counts

```python
# Fewer components
spd_config = SPDConfig(C=10)

# More components (over-parameterized)
spd_config = SPDConfig(C=50)

# Just enough components
spd_config = SPDConfig(C=tms_config.n_features)
```

### 4. Loss Coefficient Tuning

```python
# More emphasis on faithfulness
spd_config = SPDConfig(faithfulness_coeff=2.0)

# Stronger sparsity pressure
spd_config = SPDConfig(importance_minimality_coeff=1e-2)

# Different p-norm for sparsity
spd_config = SPDConfig(pnorm=2.0)  # L2 instead of L1
```

## Interpretation Guidelines

### Successful Decomposition Indicators

1. **Low Reconstruction Error**: 
   - W1 MSE < 0.01
   - W2 MSE < 0.01

2. **Sparse Causal Importance**:
   - Few components with CI > 0.1
   - Most components should have CI ≈ 0

3. **High Component Norms**:
   - Top components should have high ||U|| * ||V|| norms
   - Clear separation between active and inactive components

4. **Visual Inspection**:
   - Weight comparison plots show good reconstruction
   - Causal importance plots show clear sparsity pattern

### Troubleshooting Common Issues

**Problem**: High reconstruction error
- **Solution**: Increase training steps, reduce learning rate, or increase component count

**Problem**: No sparse causal importance
- **Solution**: Increase `importance_minimality_coeff` or try different `pnorm`

**Problem**: Training instability  
- **Solution**: Reduce learning rate, use gradient clipping, or reduce batch size

**Problem**: All components active
- **Solution**: Increase sparsity pressure or reduce model complexity

## Mathematical Analysis Tools

### Component Analysis

```python
def analyze_component_alignment(target_weights, component_U, component_V):
    """Compute alignment between components and ground truth directions"""
    reconstructed = torch.einsum('co,ci->oi', component_U, component_V)
    
    # Compute cosine similarities for each column
    cosine_sims = []
    for j in range(target_weights.size(1)):
        target_col = target_weights[:, j]
        recon_col = reconstructed[:, j]
        
        cos_sim = F.cosine_similarity(target_col, recon_col, dim=0)
        cosine_sims.append(cos_sim.item())
    
    return cosine_sims

def compute_mmcs(target_weights, components_list):
    """Compute Mean Max Cosine Similarity"""
    n_features = target_weights.size(1)
    max_sims = []
    
    for j in range(n_features):
        target_col = target_weights[:, j]
        max_sim = 0
        
        for c in range(len(components_list)):
            comp_col = components_list[c][:, j]
            sim = F.cosine_similarity(target_col, comp_col, dim=0).item()
            max_sim = max(max_sim, abs(sim))
            
        max_sims.append(max_sim)
    
    return sum(max_sims) / len(max_sims)
```

### Causal Importance Analysis

```python
def analyze_causal_importance_distribution(causal_importances, threshold=0.1):
    """Analyze distribution of causal importance values"""
    for layer_name, ci in causal_importances.items():
        mean_ci = ci.mean(dim=0)  # Average over batch
        
        active_count = (mean_ci > threshold).sum().item()
        total_count = mean_ci.size(0)
        
        print(f"{layer_name}:")
        print(f"  Active components (>{threshold}): {active_count}/{total_count}")
        print(f"  Mean CI range: [{mean_ci.min():.3f}, {mean_ci.max():.3f}]")
        print(f"  CI std: {mean_ci.std():.3f}")
        
        # Top components
        top_k = min(5, total_count)
        top_vals, top_idx = torch.topk(mean_ci, top_k)
        print(f"  Top {top_k} components: {top_idx.cpu().numpy()}")
        print(f"  Top {top_k} values: {top_vals.cpu().numpy()}")
```

## Advanced Experiments

### 1. Identity Matrix Insertion (TMS-ID)

Add identity matrices in hidden layers to test SPD's ability to handle more complex decompositions:

```python
class TMSModelWithIdentity(TMSModel):
    def __init__(self, config):
        super().__init__(config)
        # Add identity transformation in hidden layer
        self.identity = nn.Linear(config.n_hidden, config.n_hidden, bias=False)
        self.identity.weight.data = torch.eye(config.n_hidden)
        self.identity.weight.requires_grad = False  # Keep as identity
        
    def forward(self, x):
        hidden = self.linear1(x)
        hidden = self.identity(hidden)  # Identity transformation
        out_pre_relu = self.linear2(hidden)
        out = F.relu(out_pre_relu)
        return out
```

### 2. Tied Weights Experiment

```python
tms_config = TMSConfig(tied_weights=True)
# This enforces W2 = W1^T, creating symmetric decomposition
```

### 3. Different Activation Functions

```python
class TMSModelVariant(TMSModel):
    def __init__(self, config, activation='relu'):
        super().__init__(config)
        self.activation = activation
        
    def forward(self, x):
        hidden = self.linear1(x)
        out_pre_activation = self.linear2(hidden)
        
        if self.activation == 'relu':
            out = F.relu(out_pre_activation)
        elif self.activation == 'gelu':
            out = F.gelu(out_pre_activation)
        elif self.activation == 'tanh':
            out = torch.tanh(out_pre_activation)
        else:
            out = out_pre_activation
            
        return out
```

### 4. Multi-Layer Extensions

```python
class DeepTMS(nn.Module):
    def __init__(self, n_features, hidden_dims):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Build deep network
        dims = [n_features] + hidden_dims + [n_features]
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1], bias=False))
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        x = self.layers[-1](x)  # No activation on final layer
        return F.relu(x)
```

## Performance Benchmarks

### Expected Runtime (CPU)
- TMS 5-2, 2000 steps: ~2 minutes
- TMS 5-2, 10000 steps: ~8 minutes  
- TMS 40-10, 5000 steps: ~15 minutes

### Expected Results (TMS 5-2)
- **Reconstruction MSE**: < 0.01 for both W1 and W2
- **Active Components**: 2-5 components with CI > 0.1
- **Training Loss**: Converges to ~0.01 for faithfulness, ~0.1 for reconstruction
- **Component Recovery**: Should identify 5 feature-specific components

### Memory Usage
- TMS 5-2 with C=20: ~50MB
- TMS 40-10 with C=50: ~200MB
- Linear scaling with component count and layer size

## Further Reading

1. **Original SPD Paper**: "Stochastic Parameter Decomposition" by Bushnaq, Braun, and Sharkey
2. **APD Paper**: "Attribution-based Parameter Decomposition" by Braun et al.
3. **TMS Background**: "Toy Models of Superposition" by Elhage et al. (Anthropic)
4. **Mechanistic Interpretability**: Various papers on circuit discovery and feature visualization

## Contributing

To extend this implementation:

1. **New Models**: Add different target architectures in the models section
2. **New Losses**: Extend the `compute_spd_losses` function
3. **New Metrics**: Add analysis functions for different evaluation criteria
4. **Visualizations**: Enhance the plotting functions for better insights

The code is designed to be modular and extensible for research purposes.