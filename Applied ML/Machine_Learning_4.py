
#%% packages etc.

import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from our_DA_selector import da_model_selector, decision_surface_2d
from sklearn.tree import plot_tree

#%% Data Selection

data = pd.read_csv('data/class_11.csv')
y = data['G']
X = data.drop('G', axis=1)

# visualize data? 
if True:  
    df = pd.concat( (X, y), axis = 1)
    pp = sns.pairplot( df, hue = df.columns[-1], plot_kws={'alpha': 0.3})


#%% Play with DA

import numpy as np

train_list = []
test_list = []


for it in range(10):

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3)
    
    # LDA, Logit, QDA, NB, SVM, ANN, KNN, DT, RF
    
    modSpec = "svm"
    
    # For the SVC use rbf kernel as default, then go to more complicated
    # modSpec = ("svc", {"kernel":"rbf"})
    # modSpec = ("ann", {"hidden_layer_sizes":(10), "activation":"tanh", "max_iter":500, "solver":"adam"})
    
    print('*'*10, modSpec, '*'*10)
        
    model = da_model_selector(modSpec)
    model.fit(X_train, y_train)    
    
    accIS = model.score(X_train, y_train)
    accOS = model.score(X_test, y_test)
    print(f'acc_train = {accIS:.5},\t acc_test= {accOS:.5}\n')
    
    decision_surface_2d(model, X_train, y_train)
    
    train_list.append(accIS)
    test_list.append(accOS)
    
    
    
    # Decision Tree 
    # min_samples_split is when to stop
    # modSpec = ("DT", {"max_depth": 3, "min_samples_split": 40})
    # Colors tell you what the majority is
    # plot_tree(model, feature_names = list(X.columns), filled=True)

print(np.mean(train_list), np.mean(test_list))





















