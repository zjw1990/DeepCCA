import numpy as np 
import collections

def evalutateTranslation(x, y, src_words, trg_words, src_indices_test,trg_indices_test, eval_app, batch_size):
    # Build word to index map
    src2trg = collections.defaultdict(set)
    x = np.array(x, dtype='float32')
    y = np.array(y, dtype='float32')
    for src_ind,trg_ind in zip(src_indices_test,trg_indices_test):
        src2trg[src_ind].add(trg_ind)
    
    src = list(src2trg.keys())
    translation = collections.defaultdict(int)
    # nn
    if eval_app == "nn":
        for i in range(0,len(src), batch_size):
            j = min(i + batch_size,len(src))
            similarities = x[src[i:j]].dot(y.T)
            nn = similarities.argmax(axis = 1).tolist()
            for k in range(j-i):
                translation[src[i+k]] = nn[k]
    # csls
    elif eval_app == "CSLS":
  
        rt = np.zeros(y.shape[0], dtype="float32")
        #x is 200000*200
        for i in range(0,y.shape[0], batch_size):
            # compute sim with x and y
            j = min(i+batch_size,y.shape[0])
            rt[i:j] = topkMean(y[i:j].dot(x.T), k = 10, inplace=True)


        for i in range(0,len(src),batch_size):
            j = min(i+batch_size,len(src))
            # 1500*200000 sim_knn 200000
            similarities = 2*x[src[i:j]].dot(y.T) - rt  # Equivalent to the real CSLS scores for NN
            nn = similarities.argmax(axis=1).tolist()
            for k in range(j-i):
                translation[src[i+k]] = nn[k]
    
    # Compute accuracy
    accuracy = np.mean(np.array([1 if translation[i] in src2trg[i] else 0 for i in src]))
    
    return accuracy

def topkMean(m, k=10, inplace=False):
    n = m.shape[0]
    ans = np.zeros(n)
    if k <= 0:
        return ans
    if not inplace:
        m = np.array(m)
    ind0 = np.arange(n)
    ind1 = np.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        #find the maximun one
        m.argmax(axis=1, out=ind1)
        #add to ans
        ans += m[ind0, ind1]
        # replace them into the minimun one
        m[ind0, ind1] = minimum
    return ans / k