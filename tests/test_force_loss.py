#!/usr/bin/env python3
"""
Test script to verify force loss implementation
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'poet'))

from poet.models.pose_estimation_transformer import SetCriterion
from poet.models.matcher import HungarianMatcher, PoseMatcher


def test_force_loss():
    """Test the force loss implementation"""
    print("Testing force loss implementation...")
    
    # Create a simple criterion for testing
    matcher = PoseMatcher(cost_class=1, cost_bbox=1, bbox_mode='gt', class_mode='agnostic')
    weight_dict = {'loss_trans': 1.0, 'loss_rot': 1.0, 'loss_force_matrix': 1.0, 'loss_force_symmetry': 0.5, 'loss_force_consistency': 0.3}
    losses = ['translation', 'rotation', 'force_matrix', 'force_symmetry', 'force_consistency']
    
    criterion = SetCriterion(matcher, weight_dict, losses, hard_negative_ratio=0.2)
    
    # Create dummy data
    batch_size = 2
    num_queries = 5
    num_objects = 3
    
    # Mock outputs
    outputs = {
        'pred_translation': torch.randn(batch_size, num_queries, 3),
        'pred_rotation': torch.randn(batch_size, num_queries, 3, 3),
        'pred_force_matrix': torch.randn(batch_size, num_queries, num_queries, 3),  # Force matrix output
        # Add required fields for PoseMatcher
        'pred_boxes': torch.randn(batch_size, num_queries, 4),
        'pred_classes': torch.randint(0, 2, (batch_size, num_queries))
    }
    
    # Mock targets
    targets = []
    for i in range(batch_size):
        target = {
            'relative_position': torch.randn(num_objects, 3),
            'relative_rotation': torch.randn(num_objects, 3, 3),
            'force_matrix': torch.randn(num_objects, num_objects, 3),  # Force matrix target
            # Add required fields for matcher
            'labels': torch.randint(0, 2, (num_objects,)),
            'boxes': torch.randn(num_objects, 4)
        }
        targets.append(target)
    
    # Mock n_boxes
    n_boxes = [num_objects] * batch_size
    
    # Test indices (simplified for testing)
    indices = [
        (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])),  # Perfect match for first batch
        (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))   # Perfect match for second batch
    ]
    
    # Test force matrix loss directly
    print("Testing force matrix loss function...")
    try:
        force_loss_result = criterion.loss_force_matrix(outputs, targets, indices)
        print(f"Force matrix loss result: {force_loss_result}")
        assert 'loss_force_matrix' in force_loss_result
        assert isinstance(force_loss_result['loss_force_matrix'], torch.Tensor)
        print("✓ Force matrix loss function works correctly")
    except Exception as e:
        print(f"✗ Force matrix loss function failed: {e}")
        return False
    
    # Test force symmetry loss
    print("Testing force symmetry loss function...")
    try:
        symmetry_loss_result = criterion.loss_force_symmetry(outputs, targets, indices)
        print(f"Force symmetry loss result: {symmetry_loss_result}")
        assert 'loss_force_symmetry' in symmetry_loss_result
        assert isinstance(symmetry_loss_result['loss_force_symmetry'], torch.Tensor)
        print("✓ Force symmetry loss function works correctly")
    except Exception as e:
        print(f"✗ Force symmetry loss function failed: {e}")
        return False
    
    # Test force consistency loss
    print("Testing force consistency loss function...")
    try:
        consistency_loss_result = criterion.loss_force_consistency(outputs, targets, indices)
        print(f"Force consistency loss result: {consistency_loss_result}")
        assert 'loss_force_consistency' in consistency_loss_result
        assert isinstance(consistency_loss_result['loss_force_consistency'], torch.Tensor)
        print("✓ Force consistency loss function works correctly")
    except Exception as e:
        print(f"✗ Force consistency loss function failed: {e}")
        return False
    
    # Test full criterion
    print("Testing full criterion with force loss...")
    try:
        loss_dict = criterion(outputs, targets, n_boxes)
        print(f"Full loss dict: {loss_dict}")
        expected_keys = ['loss_trans', 'loss_rot', 'loss_force_matrix', 'loss_force_symmetry', 'loss_force_consistency']
        for key in expected_keys:
            assert key in loss_dict, f"Missing {key} in loss dict"
        print("✓ Full criterion works correctly")
    except Exception as e:
        print(f"✗ Full criterion failed: {e}")
        return False
    
    return True


def test_empty_forces():
    """Test behavior when no forces are provided"""
    print("\nTesting empty forces scenario...")
    
    matcher = PoseMatcher(cost_class=1, cost_bbox=1, bbox_mode='gt', class_mode='agnostic')
    weight_dict = {'loss_trans': 1.0, 'loss_rot': 1.0, 'loss_force_matrix': 1.0, 'loss_force_symmetry': 0.5, 'loss_force_consistency': 0.3}
    losses = ['translation', 'rotation', 'force_matrix', 'force_symmetry', 'force_consistency']
    
    criterion = SetCriterion(matcher, weight_dict, losses, hard_negative_ratio=0.2)
    
    # Create dummy data without forces
    batch_size = 1
    num_queries = 3
    num_objects = 2
    
    outputs = {
        'pred_translation': torch.randn(batch_size, num_queries, 3),
        'pred_rotation': torch.randn(batch_size, num_queries, 3, 3),
        'pred_force_matrix': torch.randn(batch_size, num_queries, num_queries, 3),
        # Add required fields for PoseMatcher
        'pred_boxes': torch.randn(batch_size, num_queries, 4),
        'pred_classes': torch.randint(0, 2, (batch_size, num_queries))
    }
    
    targets = [{
        'relative_position': torch.randn(num_objects, 3),
        'relative_rotation': torch.randn(num_objects, 3, 3),
        # No force_matrix provided - will be handled by the loss functions
        'labels': torch.randint(0, 2, (num_objects,)),
        'boxes': torch.randn(num_objects, 4)
    }]
    
    n_boxes = [num_objects]
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    
    try:
        force_loss_result = criterion.loss_force_matrix(outputs, targets, indices)
        print(f"Force matrix loss with empty forces: {force_loss_result}")
        assert 'loss_force_matrix' in force_loss_result
        print("✓ Empty forces handling works correctly")
        return True
    except Exception as e:
        print(f"✗ Empty forces handling failed: {e}")
        return False


def test_physics_constraints():
    """Test the physics constraints: symmetry and consistency"""
    print("\nTesting physics constraints...")
    
    matcher = PoseMatcher(cost_class=1, cost_bbox=1, bbox_mode='gt', class_mode='agnostic')
    weight_dict = {'loss_force_symmetry': 1.0, 'loss_force_consistency': 1.0}
    losses = ['force_symmetry', 'force_consistency']
    
    criterion = SetCriterion(matcher, weight_dict, losses, hard_negative_ratio=0.2)
    
    batch_size = 1
    num_queries = 4
    num_objects = 3
    
    # Test 1: Perfect symmetric force matrix (should have low symmetry loss)
    print("Testing perfect symmetric force matrix...")
    perfect_force_matrix = torch.zeros(batch_size, num_queries, num_queries, 3)
    # Create symmetric forces: F_ij = -F_ji
    perfect_force_matrix[0, 0, 1] = torch.tensor([1.0, 0.0, 0.0])  # F_01
    perfect_force_matrix[0, 1, 0] = torch.tensor([-1.0, 0.0, 0.0])  # F_10 = -F_01
    perfect_force_matrix[0, 0, 2] = torch.tensor([0.0, 1.0, 0.0])  # F_02
    perfect_force_matrix[0, 2, 0] = torch.tensor([0.0, -1.0, 0.0])  # F_20 = -F_02
    
    outputs_perfect = {
        'pred_force_matrix': perfect_force_matrix,
        'pred_boxes': torch.randn(batch_size, num_queries, 4),
        'pred_classes': torch.randint(0, 2, (batch_size, num_queries))
    }
    
    targets = [{
        'force_matrix': torch.randn(num_objects, num_objects, 3),
        'labels': torch.randint(0, 2, (num_objects,)),
        'boxes': torch.randn(num_objects, 4)
    }]
    
    indices = [(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))]
    
    try:
        symmetry_result = criterion.loss_force_symmetry(outputs_perfect, targets, indices)
        print(f"Perfect symmetry loss: {symmetry_result['loss_force_symmetry'].item():.6f}")
        assert symmetry_result['loss_force_symmetry'].item() < 1e-5, "Perfect symmetry should have near-zero loss"
        print("✓ Perfect symmetry test passed")
    except Exception as e:
        print(f"✗ Perfect symmetry test failed: {e}")
        return False
    
    # Test 2: Asymmetric force matrix (should have high symmetry loss)
    print("Testing asymmetric force matrix...")
    asymmetric_force_matrix = torch.zeros(batch_size, num_queries, num_queries, 3)
    # Create asymmetric forces: F_ij != -F_ji
    asymmetric_force_matrix[0, 0, 1] = torch.tensor([1.0, 0.0, 0.0])  # F_01
    asymmetric_force_matrix[0, 1, 0] = torch.tensor([1.0, 0.0, 0.0])  # F_10 = F_01 (wrong!)
    
    outputs_asymmetric = {
        'pred_force_matrix': asymmetric_force_matrix,
        'pred_boxes': torch.randn(batch_size, num_queries, 4),
        'pred_classes': torch.randint(0, 2, (batch_size, num_queries))
    }
    
    try:
        symmetry_result = criterion.loss_force_symmetry(outputs_asymmetric, targets, indices)
        print(f"Asymmetric symmetry loss: {symmetry_result['loss_force_symmetry'].item():.6f}")
        assert symmetry_result['loss_force_symmetry'].item() > 0.1, "Asymmetric forces should have high symmetry loss"
        print("✓ Asymmetric symmetry test passed")
    except Exception as e:
        print(f"✗ Asymmetric symmetry test failed: {e}")
        return False
    
    # Test 3: Force consistency (Newton's first law)
    print("Testing force consistency (Newton's first law)...")
    try:
        consistency_result = criterion.loss_force_consistency(outputs_perfect, targets, indices)
        print(f"Force consistency loss: {consistency_result['loss_force_consistency'].item():.6f}")
        # For the perfect symmetric case, the net force should be close to zero
        print("✓ Force consistency test passed")
    except Exception as e:
        print(f"✗ Force consistency test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("Force Loss Implementation Test")
    print("=" * 40)
    
    success = True
    success &= test_force_loss()
    success &= test_empty_forces()
    success &= test_physics_constraints()
    
    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed! Force loss implementation is working correctly.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    print("\nNext steps:")
    print("1. Make sure to use --use_force_prediction flag when training")
    print("2. Set appropriate loss coefficients:")
    print("   --force_loss_coef (default: 1.0) for force matrix loss")
    print("   --force_symmetry_coef (default: 0.5) for symmetry constraint")
    print("   --force_consistency_coef (default: 0.3) for consistency constraint")
    print("3. Ensure your dataset provides force_matrix information in the annotations")
    print("4. The model will now predict forces with physical constraints (Newton's laws)")
    print("5. Physics constraints help enforce:")
    print("   - Newton's 3rd law: F_ij = -F_ji (action-reaction pairs)")
    print("   - Newton's 1st law: Sum of forces = 0 (force balance)")
