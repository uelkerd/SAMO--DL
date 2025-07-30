    # Analyze loss pattern
    # Check training logs
    # Common causes of 0.0000 loss
    # Create test script
    # Look for training log files
    # Scenario 1: Normal case
    # Scenario 2: All zeros
    # Scenario 3: All ones
    # Scenario 4: Perfect predictions
    # Scenario 5: Very small logits
    # Suggest debugging steps
    # Summary
# Configure logging
#!/usr/bin/env python3
from pathlib import Path
import logging




"""
Simple Loss Debug Script for SAMO Deep Learning.

This script investigates the 0.0000 loss issue with minimal dependencies.
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_loss_pattern():
    """Analyze the pattern of 0.0000 loss values."""
    logger.info("🔍 Analyzing 0.0000 loss pattern...")

    causes = [
        "1. **All labels are zero** - If all target labels are 0, BCE loss can be 0",
        "2. **All labels are one** - If all target labels are 1, and predictions are perfect",
        "3. **Learning rate too high** - Model converges to trivial solution",
        "4. **Gradient explosion** - Loss becomes NaN/Inf, then gets clipped to 0",
        "5. **Loss function bug** - Incorrect loss calculation",
        "6. **Data loading issue** - Empty or corrupted batches",
        "7. **Model architecture issue** - Model produces constant outputs",
        "8. **Numerical precision** - Loss is very small but not exactly 0"
    ]

    logger.info("📋 Possible causes of 0.0000 loss:")
    for _cause in causes:
        logger.info("   {cause}")

    return causes


def check_training_logs():
    """Check for patterns in training logs."""
    logger.info("🔍 Checking training log patterns...")

    log_patterns = [
        "*.log",
        "logs/*.log",
        ".logs/*.log"
    ]

    found_logs = []
    for pattern in log_patterns:
        for log_file in Path().glob(pattern):
            found_logs.append(log_file)

    if found_logs:
        logger.info("📁 Found {len(found_logs)} log files:")
        for log_file in found_logs:
            logger.info("   {log_file}")
    else:
        logger.info("📁 No log files found")

    return found_logs


def suggest_debugging_steps():
    """Suggest debugging steps to identify the root cause."""
    logger.info("🔍 Suggesting debugging steps...")

    steps = [
        "1. **Check data distribution** - Verify labels are not all 0 or all 1",
        "2. **Monitor gradients** - Check if gradients are exploding or vanishing",
        "3. **Test with synthetic data** - Use simple test data to isolate the issue",
        "4. **Reduce learning rate** - Try 10x smaller learning rate",
        "5. **Check model outputs** - Verify model produces varied predictions",
        "6. **Test loss function** - Manually compute loss on sample data",
        "7. **Check for NaN/Inf** - Look for numerical instability",
        "8. **Verify data loading** - Ensure batches contain valid data"
    ]

    logger.info("📋 Recommended debugging steps:")
    for _step in steps:
        logger.info("   {step}")

    return steps


def create_test_script():
    """Create a simple test script to isolate the issue."""
    logger.info("🔍 Creating test script...")

    test_script = '''#!/usr/bin/env python3
"""
Simple Test Script for Loss Debugging
"""

def test_bce_loss():
    """Test BCE loss with different scenarios."""
    logging.info("🧪 Testing BCE Loss Scenarios...")

    logits = torch.randn(4, 28)  # 4 samples, 28 classes
    labels = torch.randint(0, 2, (4, 28)).float()  # Random binary labels

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    logging.info("Normal case - Loss: {loss.item():.6f}")

    logits = torch.randn(4, 28)
    labels = torch.zeros(4, 28)

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    logging.info("All zeros - Loss: {loss.item():.6f}")

    logits = torch.randn(4, 28)
    labels = torch.ones(4, 28)

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    logging.info("All ones - Loss: {loss.item():.6f}")

    logits = torch.tensor([[10.0, -10.0, 10.0, -10.0]] * 4)  # Strong predictions
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]] * 4)  # Perfect targets

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    logging.info("Perfect predictions - Loss: {loss.item():.6f}")

    logits = torch.tensor([[0.001, -0.001, 0.001, -0.001]] * 4)
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]] * 4)

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    logging.info("Small logits - Loss: {loss.item():.6f}")

if __name__ == "__main__":
    test_bce_loss()
'''

    with open("scripts/test_loss_scenarios.py", "w") as f:
        f.write(test_script)

    logger.info("✅ Created test script: scripts/test_loss_scenarios.py")
    return "scripts/test_loss_scenarios.py"


def main():
    """Main debugging function."""
    logger.info("🚀 Starting simple loss debugging...")

    analyze_loss_pattern()

    check_training_logs()

    suggest_debugging_steps()

    create_test_script()

    logger.info("\n" + "="*60)
    logger.info("📋 SIMPLE DEBUG SUMMARY")
    logger.info("="*60)

    logger.info("🎯 Most likely causes of 0.0000 loss:")
    logger.info("   1. All labels are zero (most common)")
    logger.info("   2. Learning rate too high causing convergence to trivial solution")
    logger.info("   3. Model architecture producing constant outputs")
    logger.info("   4. Loss function implementation bug")

    logger.info("\n🔧 Immediate actions to take:")
    logger.info("   1. Run: python scripts/test_loss_scenarios.py")
    logger.info("   2. Check your training data labels")
    logger.info("   3. Reduce learning rate by 10x")
    logger.info("   4. Add gradient monitoring to training loop")

    logger.info("\n⚠️  CRITICAL: 0.0000 loss indicates training is not working!")
    logger.info("   This needs immediate attention before continuing training.")

    logger.info("🔍 Simple debugging complete!")


if __name__ == "__main__":
    main()
