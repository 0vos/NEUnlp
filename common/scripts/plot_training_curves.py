import json
import matplotlib.pyplot as plt
import sys

def plot_trainer_state(json_path, out_path):
    with open(json_path) as f:
        data = json.load(f)
    
    logs = data.get('log_history', [])
    
    # 分离 train 和 eval
    train_steps, train_loss, grad_norm = [], [], []
    eval_steps, eval_loss = [], []
    
    for entry in logs:
        if 'loss' in entry and 'step' in entry:
            train_steps.append(entry['step'])
            train_loss.append(entry['loss'])
            if 'grad_norm' in entry:
                grad_norm.append(entry['grad_norm'])
        if 'eval_loss' in entry and 'step' in entry:
            eval_steps.append(entry['step'])
            eval_loss.append(entry['eval_loss'])
    
    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    if train_steps:
        axes[0].plot(train_steps, train_loss, 'b-', label='Train Loss')
    if eval_steps:
        axes[0].plot(eval_steps, eval_loss, 'r-', marker='o', label='Eval Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Eval Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Grad Norm
    if grad_norm:
        axes[1].plot(train_steps[:len(grad_norm)], grad_norm, 'g-', label='Grad Norm')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Grad Norm')
        axes[1].set_title('Gradient Norm')
        axes[1].grid(True, alpha=0.3)
    
    # Learning Rate (if available)
    lr_steps, lrs = [], []
    for entry in logs:
        if 'learning_rate' in entry and 'step' in entry:
            lr_steps.append(entry['step'])
            lrs.append(entry['learning_rate'])
    if lrs:
        axes[2].plot(lr_steps, lrs, 'm-', label='LR')
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')

if __name__ == '__main__':
    plot_trainer_state(sys.argv[1], sys.argv[2])
