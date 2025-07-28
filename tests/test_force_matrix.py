"""
Test script for Force Matrix Head functionality
Tests the new pairwise force prediction capabilities
"""

import torch
import torch.nn.functional as F
import numpy as np
from models.force_matrix_head import ForceMatrixHead

def test_force_matrix_head():
    """Test ForceMatrixHead forward pass"""
    print("🧪 Testing ForceMatrixHead...")
    
    # Create model
    in_dim = 256
    hidden_dim = 256
    num_heads = 8
    num_layers = 2
    
    force_head = ForceMatrixHead(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        edge_features='sum'
    )
    
    # Create dummy input
    batch_size = 2
    num_queries = 5
    
    # Object features from PoET decoder
    hs = torch.randn(batch_size, num_queries, in_dim)
    
    print(f"Input shape: {hs.shape}")
    
    # Forward pass
    pred_force_matrix = force_head(hs)
    
    print(f"Output shape: {pred_force_matrix.shape}")
    print(f"Expected shape: [{batch_size}, {num_queries}, {num_queries}, 3]")
    
    # Verify output shape
    expected_shape = (batch_size, num_queries, num_queries, 3)
    assert pred_force_matrix.shape == expected_shape, f"Shape mismatch: {pred_force_matrix.shape} vs {expected_shape}"
    
    print("✅ ForceMatrixHead test passed!")
    return pred_force_matrix

def test_loss_function():
    """Test force matrix loss function"""
    print("\n🧪 Testing force matrix loss function...")
    
    # Import required modules
    from models.pose_estimation_transformer import SetCriterion
    from models.matcher import HungarianMatcher
    
    # Create dummy matcher
    matcher = HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=1)
    
    # Create criterion
    weight_dict = {'loss_force_matrix': 1.0}
    losses = ['force_matrix']
    criterion = SetCriterion(matcher, weight_dict, losses, hard_negative_ratio=0.2)
    
    # Create dummy outputs and targets
    batch_size = 2
    num_queries = 4
    
    # Dummy predictions
    outputs = {
        'pred_force_matrix': torch.randn(batch_size, num_queries, num_queries, 3)
    }
    
    # Dummy targets
    targets = []
    for i in range(batch_size):
        n_objects = 3  # Number of real objects in this image
        
        # Create force matrix for this image
        force_matrix = torch.zeros(n_objects, n_objects, 3)
        # Add some non-zero forces
        force_matrix[0, 1] = torch.tensor([1.0, 0.5, 0.2])  # Force from object 0 to object 1
        force_matrix[1, 2] = torch.tensor([0.3, 1.2, 0.8])  # Force from object 1 to object 2
        
        target = {
            'force_matrix': force_matrix,
            'labels': torch.tensor([1, 2, 3]),  # Object classes
            'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1], [0.3, 0.3, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1]])
        }
        targets.append(target)
    
    # Create dummy indices (Hungarian matching results)
    indices = []
    for i in range(batch_size):
        src_idx = torch.tensor([0, 1, 2])  # Predicted object indices
        tgt_idx = torch.tensor([0, 1, 2])  # Target object indices
        indices.append((src_idx, tgt_idx))
    
    # Test loss computation
    try:
        loss_result = criterion.loss_force_matrix(outputs, targets, indices)
        print(f"Loss result: {loss_result}")
        
        assert 'loss_force_matrix' in loss_result, "Loss key not found"
        assert torch.is_tensor(loss_result['loss_force_matrix']), "Loss is not a tensor"
        assert loss_result['loss_force_matrix'].item() >= 0, "Loss should be non-negative"
        
        print("✅ Force matrix loss test passed!")
        return loss_result
        
    except Exception as e:
        print(f"❌ Loss test failed: {e}")
        raise e

def test_data_format():
    """Test expected data format for force matrices"""
    print("\n🧪 Testing force matrix data format...")
    
    # Example of expected force matrix format
    n_objects = 3
    force_matrix = torch.zeros(n_objects, n_objects, 3)
    
    # Force from object 0 to object 1: [fx, fy, fz]
    force_matrix[0, 1] = torch.tensor([1.5, 0.8, 0.2])
    
    # Force from object 1 to object 2: [fx, fy, fz]
    force_matrix[1, 2] = torch.tensor([0.3, 1.2, 0.9])
    
    # Force from object 2 to object 0: [fx, fy, fz]
    force_matrix[2, 0] = torch.tensor([-0.5, -0.8, 0.1])
    
    print(f"Force matrix shape: {force_matrix.shape}")
    print(f"Force from object 0 to 1: {force_matrix[0, 1]}")
    print(f"Force from object 1 to 2: {force_matrix[1, 2]}")
    print(f"Force from object 2 to 0: {force_matrix[2, 0]}")
    
    # Verify anti-symmetry property (optional, for Newton's third law)
    print(f"Force 0->1: {force_matrix[0, 1]}")
    print(f"Force 1->0: {force_matrix[1, 0]}")
    print("Note: Newton's third law would require force_matrix[i,j] = -force_matrix[j,i]")
    
    print("✅ Data format test completed!")

def main():
    """Run all tests"""
    print("🚀 Starting Force Matrix Prediction Tests")
    print("=" * 50)
    
    try:
        # Test 1: ForceMatrixHead functionality
        pred_matrix = test_force_matrix_head()
        
        # Test 2: Loss function
        loss_result = test_loss_function()
        
        # Test 3: Data format
        test_data_format()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Force matrix prediction is ready!")
        print("\nNext steps:")
        print("1. Prepare dataset with force_matrix annotations")
        print("2. Update training configuration")
        print("3. Start training with force prediction enabled")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise e

if __name__ == "__main__":
    main()
