"""
Quick optimizations to improve current model performance
"""

# 1. Increase learning rate slightly for faster convergence
# Change in modelclaude.py line 1159:
# lr=0.00001 → lr=0.00005

# 2. Reduce patience for early stopping
# Change line 1191:
# patience=35 → patience=15

# 3. Add learning rate scheduler
# After optimizer definition, add:
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',           # Monitor F1 score (maximize)
    factor=0.8,           # Reduce LR by 20%
    patience=5,           # Wait 5 epochs
    min_lr=1e-6          # Minimum LR
)

print("Quick optimizations:")
print("1. LR: 0.00001 → 0.00005 (5x increase)")
print("2. Patience: 35 → 15 (stop sooner)")
print("3. Add ReduceLROnPlateau scheduler")
print("4. Target F1: 0.65-0.70")