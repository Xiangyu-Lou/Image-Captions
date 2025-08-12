import  matplotlib.pyplot as plt
import json

with open('models/log.json', 'r') as f:
    log = json.load(f)
    
plt.plot(range(len(log['all_train_loss'][16:])), log['all_train_loss'][16:], label='Train Loss', color='orange')
plt.plot(range(len(log['all_train_loss'][16:])), log['all_eval_loss'][16:], label='Evaluate Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()