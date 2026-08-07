import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

df3 = pd.read_csv('dataset_part_3.csv')  # already one-hot encoded features
df2 = pd.read_csv('spacex_data_step2_enriched.csv')  # has Class label

Y = df2['Class'].to_numpy()
X = StandardScaler().fit_transform(df3.to_numpy())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print("Training set size:", X_train.shape[0], " Test set size:", X_test.shape[0])

models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000),
        {'C': [0.01, 0.1, 1], 'penalty': ['l2'], 'solver': ['lbfgs']}),
    'SVM': (SVC(),
        {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale']}),
    'Decision Tree': (DecisionTreeClassifier(random_state=2),
        {'max_depth': [2, 4, 6, None], 'criterion': ['gini', 'entropy']}),
    'KNN': (KNeighborsClassifier(),
        {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}),
}

results = {}
best_overall = None
best_score = -1

for name, (model, params) in models.items():
    gs = GridSearchCV(model, params, cv=5, scoring='accuracy')
    gs.fit(X_train, Y_train)
    test_acc = gs.score(X_test, Y_test)
    results[name] = {'best_params': gs.best_params_, 'cv_score': gs.best_score_, 'test_acc': test_acc, 'estimator': gs.best_estimator_}
    print(f"\n{name}: best_params={gs.best_params_}, CV acc={gs.best_score_:.4f}, Test acc={test_acc:.4f}")
    if test_acc > best_score:
        best_score = test_acc
        best_overall = name

print(f"\n=== Best model: {best_overall} (test accuracy {best_score:.4f}) ===")

# Confusion matrix for the best model
best_model = results[best_overall]['estimator']
Y_pred = best_model.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Did not land', 'Landed'])
disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
ax.set_title(f'Confusion Matrix — {best_overall} (Best Model)', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix_best_model.png', dpi=150)
plt.close()

# Model comparison bar chart
fig, ax = plt.subplots(figsize=(9, 6))
names = list(results.keys())
accs = [results[n]['test_acc'] for n in names]
cv_accs = [results[n]['cv_score'] for n in names]
x = np.arange(len(names))
width = 0.35
ax.bar(x - width/2, cv_accs, width, label='CV Accuracy', color='#1f77b4')
ax.bar(x + width/2, accs, width, label='Test Accuracy', color='#2ca02c')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15)
ax.set_ylabel('Accuracy', fontsize=13)
ax.set_title('Model Comparison — CV vs. Test Accuracy', fontsize=14)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.close()

print("\nSaved confusion_matrix_best_model.png and model_comparison.png")

with open('model_results_summary.txt', 'w') as f:
    f.write(f"Best model: {best_overall}\nTest accuracy: {best_score:.4f}\n\n")
    for n, r in results.items():
        f.write(f"{n}: best_params={r['best_params']}, CV acc={r['cv_score']:.4f}, Test acc={r['test_acc']:.4f}\n")
