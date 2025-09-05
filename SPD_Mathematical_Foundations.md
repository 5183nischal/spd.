# Stochastic Parameter Decomposition (SPD) - Mathematical Foundations and Implementation

## Abstract

This document provides a comprehensive mathematical explanation of Stochastic Parameter Decomposition (SPD), a novel method for decomposing neural network parameters into interpretable, sparse rank-one components. SPD addresses the limitations of Attribution-based Parameter Decomposition (APD) by introducing stochastic masking and learned causal importance functions, making it more scalable and robust to hyperparameters.

## 1. Mathematical Framework

### 1.1 Problem Setup

Consider a trained neural network $f(x; W)$ that maps inputs $x$ to outputs $y = f(x; W)$, parameterized by weight matrices $W = \{W^1, \ldots, W^L\}$ where $L$ is the number of layers.

SPD aims to decompose each weight matrix $W^l \in \mathbb{R}^{d_{out} \times d_{in}}$ into a sum of rank-one components:

$$W^l_{i,j} \approx \sum_{c=1}^C U^l_{i,c} V^l_{c,j}$$

where:
- $U^l_c \in \mathbb{R}^{d_{out}}$ and $V^l_c \in \mathbb{R}^{d_{in}}$ are rank-one component vectors
- $C$ is the number of components per layer (can exceed the matrix rank)
- $c$ indexes the component

### 1.2 Causal Importance Functions

SPD introduces causal importance functions $\Gamma^l_c: X \rightarrow [0,1]$ that predict how much each component $c$ in layer $l$ can be ablated for a given input $x$:

$$g^l_c(x) = \Gamma^l_c(x) = \sigma_H(\gamma^l_c(h^l_c(x)))$$

where:
- $h^l_c(x) = \sum_j V^l_{c,j} a^l_j(x)$ is the "inner activation" (dot product of component $V^l_c$ with pre-activation $a^l(x)$)
- $\gamma^l_c$ is a small MLP (typically 1-2 hidden layers)  
- $\sigma_H$ is the hard sigmoid: $\sigma_H(z) = \text{clip}((z+1)/2, 0, 1)$

### 1.3 Stochastic Masking

The key innovation of SPD is stochastic masking. For each component and input, we sample stochastic masks:

$$m^l_c(x, r) = g^l_c(x) + (1 - g^l_c(x)) r^l_c$$

where $r^l_c \sim \mathcal{U}(0, 1)$ is a uniform random variable.

This creates masked weight matrices:
$$W'^l_{i,j}(x, r) = \sum_{c=1}^C U^l_{i,c} \cdot m^l_c(x, r) \cdot V^l_{c,j}$$

The masked network becomes: $f(x | W'^1(x,r), \ldots, W'^L(x,r))$

## 2. Loss Functions

SPD optimizes four loss terms:

### 2.1 Faithfulness Loss

Ensures components sum to original parameters:
$$\mathcal{L}_{\text{faithfulness}} = \frac{1}{N} \sum_{l=1}^L \sum_{i,j} \left(W^l_{i,j} - \sum_{c=1}^C U^l_{i,c} V^l_{c,j}\right)^2$$

where $N$ is the total number of parameters.

### 2.2 Stochastic Reconstruction Loss  

Trains the masked network to match the target output:
$$\mathcal{L}_{\text{stochastic-recon}} = \frac{1}{S} \sum_{s=1}^S D\left(f(x | W'(x, r^{(s)})), f(x | W)\right)$$

where:
- $S$ is the number of mask samples per training step
- $D$ is a divergence measure (MSE for regression, KL divergence for classification)
- $r^{(s)}$ are independent random samples

### 2.3 Stochastic Reconstruction Layerwise Loss

An auxiliary loss that masks only one layer at a time:
$$\mathcal{L}_{\text{stochastic-recon-layerwise}} = \frac{1}{LS} \sum_{l=1}^L \sum_{s=1}^S D\left(f(x | W^1, \ldots, W'^l(x, r^{l,(s)}), \ldots, W^L), f(x | W)\right)$$

This reduces noise compared to masking all layers simultaneously.

### 2.4 Importance Minimality Loss

Encourages causal importance values to be small (components to be ablatable):
$$\mathcal{L}_{\text{importance-minimality}} = \sum_{l=1}^L \sum_{c=1}^C |g^l_c(x)|^p$$

where $p > 0$ is typically 1 or 2.

### 2.5 Combined Loss

The total SPD loss is:
$$\mathcal{L}_{\text{SPD}} = \mathcal{L}_{\text{faithfulness}} + \beta_1 \mathcal{L}_{\text{stochastic-recon}} + \beta_2 \mathcal{L}_{\text{stochastic-recon-layerwise}} + \beta_3 \mathcal{L}_{\text{importance-minimality}}$$

## 3. Toy Model of Superposition (TMS)

### 3.1 TMS Architecture

The TMS model has the form:
$$\hat{x} = \text{ReLU}(W_2 W_1 x + b)$$

where:
- $W_1 \in \mathbb{R}^{m_1 \times m_2}$ (compression: features → hidden)
- $W_2 \in \mathbb{R}^{m_2 \times m_1}$ (decompression: hidden → features)  
- $m_1 < m_2$ creates a bottleneck that forces superposition
- Input $x$ is sparse with features activating with probability $p$

### 3.2 Ground Truth Mechanisms

In TMS, the ground truth mechanisms are rank-1 matrices corresponding to individual feature directions. For feature $i$, the mechanism should be:
- A matrix that is zero everywhere except column $i$
- Column $i$ contains the corresponding column from $W_1$ or $W_2$

### 3.3 TMS Training

TMS is trained with importance-weighted reconstruction loss:
$$\mathcal{L}_{\text{TMS}} = \mathbb{E}_{x} \left[ \lambda \cdot (|x| - \hat{x})^2 \right]$$

where $\lambda$ is the importance weighting (typically 1.0).

## 4. Implementation Details

### 4.1 Causal Importance MLP Architecture

Each $\gamma^l_c$ is implemented as:
```
Input: h^l_c(x) ∈ ℝ (scalar)
Hidden: Linear(1, hidden_dim) → GELU → ... → Linear(hidden_dim, 1)  
Output: Hard Sigmoid → g^l_c(x) ∈ [0,1]
```

Typical hidden dimensions: [16] or [32, 16]

### 4.2 Training Algorithm

```python
for step in range(training_steps):
    # 1. Sample batch
    x, y = sample_batch()
    
    # 2. Compute target output
    target_out = target_model(x)
    
    # 3. Compute causal importances
    causal_importances = {}
    for layer in component_layers:
        ci = component_model.compute_causal_importances(layer, x)
        causal_importances[layer] = ci
    
    # 4. Sample stochastic masks
    masks = sample_stochastic_masks(causal_importances, n_samples=S)
    
    # 5. Compute losses
    faithfulness_loss = compute_faithfulness_loss(component_model)
    
    stoch_recon_loss = 0
    for mask in masks:
        masked_out = component_model.forward_with_masks(x, mask)
        stoch_recon_loss += mse_loss(masked_out, target_out)
    stoch_recon_loss /= len(masks)
    
    stoch_recon_layerwise_loss = 0
    for layer in component_layers:
        for mask in masks:
            layerwise_out = component_model.forward_layerwise_masked(x, layer, mask[layer])
            stoch_recon_layerwise_loss += mse_loss(layerwise_out, target_out)
    stoch_recon_layerwise_loss /= (len(component_layers) * len(masks))
    
    importance_min_loss = 0
    for layer, ci in causal_importances.items():
        importance_min_loss += torch.sum(torch.abs(ci) ** p)
    importance_min_loss /= batch_size
    
    # 6. Combine and optimize
    total_loss = (faithfulness_coeff * faithfulness_loss + 
                  stoch_recon_coeff * stoch_recon_loss +
                  stoch_recon_layerwise_coeff * stoch_recon_layerwise_loss +
                  importance_min_coeff * importance_min_loss)
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

### 4.3 Hyperparameters

Based on the paper and successful configurations:

- **Component count**: $C = 20$ (for TMS 5-2), $C = 50$ (for larger models)
- **Mask samples**: $S = 1$ (sufficient in practice)
- **Loss coefficients**:
  - $\beta_1 = 1.0$ (stochastic reconstruction)
  - $\beta_2 = 1.0$ (stochastic reconstruction layerwise)  
  - $\beta_3 = 3 \times 10^{-3}$ (importance minimality)
- **p-norm**: $p = 1.0$
- **Learning rate**: $10^{-3}$
- **Training steps**: 40,000 (paper), 5,000+ (practical)

## 5. Advantages over APD

### 5.1 Scalability
- **Memory**: SPD uses rank-1 components vs. full parameter components in APD
- **Computation**: No expensive attribution computations required
- **Components**: Can handle $C >> \min(d_{in}, d_{out})$ for superposition

### 5.2 Robustness  
- **Hyperparameters**: No need to tune top-$k$ selection
- **Training**: Stochastic masking provides natural regularization
- **Convergence**: More stable optimization landscape

### 5.3 Interpretability
- **Direct ablation**: Causal importance directly measures ablatability
- **Sparse activation**: Importance minimality encourages sparse component usage
- **Ground truth recovery**: Better mechanism identification in toy models

## 6. Experimental Validation

### 6.1 Metrics

**Mean Max Cosine Similarity (MMCS)**:
$$\text{MMCS}(W, \{U_c V_c^T\}) = \frac{1}{m_2} \sum_{j=1}^{m_2} \max_c \frac{U_{:,c} V_{c,j} \cdot W_{:,j}}{||U_{:,c} V_{c,j}||_2 ||W_{:,j}||_2}$$

**Mean L2 Ratio (ML2R)**:
$$\text{ML2R}(W, \{U_c V_c^T\}) = \frac{1}{m_2} \sum_{j=1}^{m_2} \frac{||U_{:,c^*} V_{c^*,j}||_2}{||W_{:,j}||_2}$$

where $c^* = \arg\max_c \text{cosine similarity}(U_{:,c} V_{c,j}, W_{:,j})$

### 6.2 Expected Results

For successful SPD decomposition:
- **MMCS ≈ 1.0**: Components align with ground truth directions
- **ML2R ≈ 1.0**: Component magnitudes match ground truth (no shrinkage)
- **Active components**: Few components with high causal importance (> 0.1)
- **Reconstruction error**: Low MSE between $W$ and $\sum_c U_c V_c^T$

## 7. Extensions and Future Work

### 7.1 Clustering Components
For unknown ground truth, algorithmic clustering of rank-1 components into full mechanisms:
- Correlation-based clustering of activation patterns
- Spectral clustering on component similarity matrices
- Hierarchical clustering with mechanistic similarity metrics

### 7.2 Multi-layer Components
Extend to components spanning multiple layers:
- Cross-layer causal importance functions
- Attention mechanisms for component interactions
- Compositional component hierarchies

### 7.3 Scaling to Large Models
- Gradient checkpointing for memory efficiency
- Distributed training across components
- Progressive component discovery

## 8. Conclusion

SPD provides a mathematically principled and practically effective approach to neural network decomposition. By combining rank-one parameter decomposition with learned causal importance functions and stochastic masking, SPD overcomes the key limitations of previous methods while maintaining interpretability and faithfulness to the original network.

The method's success on toy models demonstrates its potential for understanding neural network mechanisms, with clear paths for scaling to larger, real-world models through the extensions outlined above.