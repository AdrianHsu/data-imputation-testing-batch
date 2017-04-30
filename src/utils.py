import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict


def load_data(data_path,data_name,cat_feature):
    data = pd.read_csv(data_path,header=None)
    data = data.reindex(np.random.permutation(data.index))
    data = data.reset_index(drop=True)
    y = data.iloc[:,-1]
    x = data.iloc[:,0:-1]

    feature_numerical = [f for f in range(x.shape[1]) if f not in cat_feature]
    return x,y,feature_numerical

def preprocessing_train(x_train, feature_numerical, feature_cat):
    
    x_train.loc[:,feature_numerical] = scale(x_train.loc[:,feature_numerical])
    x_train = pd.get_dummies(x_train,columns=feature_cat)
    
    return x_train

def Calculate_feature_imp(x_train,y_train):
    clf = RandomForestClassifier(random_state=True)
    clf.fit(x_train.values,y_train.values)
    feature_imp_score = np.argsort(clf.feature_importances_)[::-1]
    tmp = int(len(feature_imp_score)/2.0)
    feature_imp = feature_imp_score[0:tmp]
    feature_notimp = feature_imp_score[tmp:]
    
    return feature_imp,feature_notimp

def preprocessing_test(x_test_original,feature_imp,feature_cat,feature_numerical,missing_imp_prob):
    X_mask = np.ones(x_test_original.shape)
        
    for col in xrange(X_mask.shape[1]):

        if col in feature_imp:
            nan_idx = np.random.choice(X_mask.shape[0],int(X_mask.shape[0] * missing_imp_prob),replace=False)
            X_mask[[nan_idx],col] = np.nan
        else:
            nan_idx = np.random.choice(X_mask.shape[0],int(X_mask.shape[0] * (1.0 - missing_imp_prob)),replace=False)
            X_mask[[nan_idx],col] = np.nan

    x_test_missing = x_test_original * X_mask
    X_mask = np.nan_to_num(X_mask)    
    #dummy_na = True we create another colum to specify whether a specific row has missing value in specific cat feature
    # For example if col 2 is categorical value in ['A','B'], if entry(1,2) is NaN then the value in 2_na will be 1
    x_test_missing = pd.get_dummies(x_test_missing,columns=feature_cat,dummy_na=True)

    X_mask = pd.DataFrame(X_mask)
    
    X_mask.drop(feature_cat,axis=1,inplace=True)
    X_mask = pd.concat([X_mask,x_test_missing.ix[:,len(feature_numerical):]],axis=1)
    
    return x_test_missing,X_mask

def MissingMaskCreation(mask,feature_numerical):
    
    mask_cat = mask.ix[:,len(feature_numerical):]
    invert_nan = defaultdict(list)
    for c in mask_cat.columns:
        if 'nan' in c:
            mask_cat[c] = mask_cat[c].apply(lambda x : 0 if x==1 else 1)
            invert_nan[c] = mask_cat[c]
    for c in mask_cat.columns:

        cat = c.split("_")[0]
        mask_cat[c] += invert_nan[cat + "_nan"]
        mask_cat[c] = mask_cat[c].apply(lambda x: 1 if x==2 else x)
    return pd.concat([mask.ix[:,0:len(feature_numerical)],mask_cat],axis = 1)

def CleanCatNan(X,feature_cat):
    catnan = []
    for cf in feature_cat:
        catnan.append("_".join([str(cf),'nan']))
    
    X = X.drop(catnan,1)
    #for f in X.columns:
    #    if isinstance(f,str):
    X.columns = [f.split("_")[0] + "_" + str(int(float(f.split("_")[1]))) if isinstance(f,str) else f for f in X.columns]        
    return X

def RemoveUnseenFeature(tr,tt):
    
    tr_f = set(tr.columns)
    tt_f = set(tt.columns)
    
    feature_union = tr_f.union(tt_f)
    feature_intersect = tr_f.intersection(tt_f)
    unseen = list(feature_union.difference(feature_intersect))
    print "Unseen feature in test: "
    print unseen
    for un in unseen:
        if un in tt_f:
            tt.drop(un,axis=1,inplace=True)
        elif un in tr_f:
            tr.drop(un,axis=1,inplace=True)
    return tr,tt,unseen

def Normalization(X,feature_numerical):
    

    X = X.astype(float)
    X = X.values
    for i in range(len(feature_numerical)):
        temp = np.array([[value] for value in X[:,i] if not np.isnan(value)])
        if len(temp) > 0:
            temp = scale(temp)           
            flag = 0
            for j in range(X.shape[0]):
                if not np.isnan(X[j,i]):
                    X[j,i] = temp[flag,0]
                    flag += 1

    return X

def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]

def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]
