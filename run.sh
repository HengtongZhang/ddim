# Train unconditional CIFAR 10.
python main.py --config cifar10.yml --exp ddim_exp --doc logs --ni
# python main.py --config cifar10.yml --exp ddim_exp --doc logs --ni --resume_training

# Train conditional CIFAR 10.
python main.py --config cifar10.yml --exp ddim_exp --doc logs --ni --cond
# python main.py --config cifar10.yml --exp ddim_exp --doc logs --ni --cond --resume_training

# Train conditional Celeba.
python main.py --config celaba.yml --exp ddim_exp --doc logs --ni --cond
# python main.py --config celaba.yml --exp ddim_exp --doc logs --ni --cond --resume_training