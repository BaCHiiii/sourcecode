from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# y_true: ground truth (0 or 1)
# y_scores: 每個樣本的異常分數（越高越可能是後門）
tpr = [0,0.979,1]
fpr =[0,0.195,1]
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUROC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Backdoor Detection')
plt.legend()
plt.grid(True)
plt.show()
