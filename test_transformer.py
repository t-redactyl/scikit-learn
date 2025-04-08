"""Test script for GPTTransformer.

This script demonstrates how to use the GPTTransformer class with both
custom architecture and pretrained models from Hugging Face.
"""

import numpy as np
import os
from sklearn.neural_network import GPTTransformer

# Create a simple dataset
X = np.random.rand(10, 20)  # 10 samples, 20 features

print("=" * 50)
print("Testing GPTTransformer with custom architecture")
print("=" * 50)

# Create and fit the transformer with custom architecture
transformer_custom = GPTTransformer(
    n_layers=2,  # Using fewer layers for faster execution
    n_heads=4,
    embedding_dim=64,
    verbose=1
)

# Fit the transformer to the data
transformer_custom.fit(X)

# Transform the data
X_transformed_custom = transformer_custom.transform(X)

print(f"Input shape: {X.shape}")
print(f"Output shape: {X_transformed_custom.shape}")
print(f"Transformer parameters: {transformer_custom.get_params()}")

# Check if transformers package is installed
try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    has_transformers = True
except ImportError:
    has_transformers = False

if has_transformers:
    print("\n" + "=" * 50)
    print("Testing GPTTransformer with pretrained model from Hugging Face")
    print("=" * 50)

    # Create and fit the transformer with pretrained model
    # Using a small model for faster execution
    transformer_pretrained = GPTTransformer(
        use_pretrained=True,
        pretrained_model_name="distilgpt2",  # Small GPT-2 model
        output_hidden_states=True,
        verbose=1
    )

    # Fit the transformer to the data
    transformer_pretrained.fit(X)

    # Transform the data
    X_transformed_pretrained = transformer_pretrained.transform(X)

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {X_transformed_pretrained.shape}")
    print(f"Transformer parameters: {transformer_pretrained.get_params()}")

    print("\n" + "=" * 50)
    print("Testing GPTTransformer with fine-tuning of pretrained model")
    print("=" * 50)

    # Create and fit the transformer with fine-tuning enabled
    # Using a small model for faster execution and fewer iterations for demonstration
    transformer_finetuned = GPTTransformer(
        use_pretrained=True,
        pretrained_model_name="distilgpt2",  # Small GPT-2 model
        fine_tune=True,  # Enable fine-tuning
        output_hidden_states=True,
        n_iter=2,  # Using fewer iterations for faster execution
        batch_size=5,
        learning_rate=0.0001,  # Smaller learning rate for fine-tuning
        verbose=1
    )

    # Fit the transformer to the data (this will fine-tune the model)
    transformer_finetuned.fit(X)

    # Transform the data
    X_transformed_finetuned = transformer_finetuned.transform(X)

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {X_transformed_finetuned.shape}")
    print(f"Transformer parameters: {transformer_finetuned.get_params()}")
else:
    print("\n" + "=" * 50)
    print("Skipping pretrained model test - transformers package not installed")
    print("To install: pip install transformers torch")
    print("=" * 50)
