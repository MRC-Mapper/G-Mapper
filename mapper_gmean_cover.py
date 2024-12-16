import numpy as np
from scipy.stats import anderson
from statistics import stdev
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from mapper import Cover, generate_mapper_graph
import copy
import time
"""gmeans_cover.
-------
Parameters
----------
X: array-like of shape (n_samples, n_features)
lens : array-like of shape (n_samples, 1)
iterations: the number of iterations, default=10
max_intervals: the maximum number of intervals, default=20
method: {'BFS, 'randomized', 'DFS'}, default='BFS'
ad_threshold: the threshold for Anderson-Darling test statistic, default=10
g_overlap: the number representing how two intervals intersect, default=0.1
initial_cover: the initial cover, default=the whole space
-------
Returns
-------
mapper
"""
    
def ad_test(data):
    # Anderson-Darling test.
    n=len(data)
    and_corrected = anderson(data)[0]*(1+4/n-25/(n^2))
    return and_corrected

def gm_split(interval,membership,g_overlap):
    # l: interval, L: membership, c: mean, std: standard deviation.
    I=interval
    L=membership
    c = np.mean(L)
    std = np.std(L, ddof=1)
    m = np.sqrt(2/np.pi)*std 
    c1 = c+m
    c2 = c-m
    L=L.reshape(-1,1)
    # apply gmm.
    gmm = GaussianMixture(n_components=2, means_init=[[c1], [c2]], covariance_type='full').fit(L)
    # left component.
    left_index=np.argmin(gmm.means_)
    left_mean=np.min(gmm.means_)
    left_std=np.sqrt(gmm.covariances_[left_index])[0][0]
    # right component.
    right_index=np.argmax(gmm.means_)
    right_mean=np.max(gmm.means_)
    right_std=np.sqrt(gmm.covariances_[right_index])[0][0]
    # new interval=[left interval, right_interval].
    left_interval=[I[0],np.min([left_mean+(1+g_overlap)*left_std/(left_std+right_std)*(right_mean-left_mean),I[1]])]
    right_interval=[np.max([right_mean-(1+g_overlap)*right_std/(left_std+right_std)*(right_mean-left_mean),I[0]]),I[1]]
    new_interval=np.array([left_interval,right_interval])
    return new_interval




def membership(data,intervals):
    # assign the merbership for intervals.
    membership=[[] for i in range(len(intervals))]
    for i in range(len(data)):
        for j in range(len(intervals)):
            if (data[i]>=intervals[j][0]) and (data[i]<=intervals[j][1]):
                membership[j].append(data[i])         
    return membership

def split(interval_membership,cover,g_overlap,index):
    # split the j-th interval.
    j=index
    split_interval=gm_split(cover.intervals[j],np.array(interval_membership[j]),g_overlap)
    new_membership=membership(interval_membership[j],split_interval)
    cover.intervals=np.delete(cover.intervals,j,axis=0)
    cover.intervals=np.insert(cover.intervals, j, split_interval, axis=0)
    interval_membership.pop(j)
    interval_membership.insert(j,new_membership[0]) 
    interval_membership.insert(j+1,new_membership[1]) 
    
def gmeans_cover(X, lens, iterations=10, max_intervals=20, method=None, ad_threshold=10, g_overlap=0.1, initial_cover=Cover()): 
    cover=copy.deepcopy(initial_cover)
    if (str(type(cover.intervals))=="<class 'NoneType'>"):
        cover.compute_intervals(np.min(lens), np.max(lens), lens)
    num_iter = 0
    check_interval = [True for i in range(len(cover.intervals))]
    # interval_membership: the elements which belong to the i-th interval.
    interval_membership=membership(lens,cover.intervals)
    ad_scores=[]
    split_index=[]
    # iterations.
    for iteration in range(iterations):
        modified = False
        if method is None or method == 'BFS':
            if len(ad_scores)==0:
                for i in range(len(cover.intervals)):
                    if not check_interval[i]:
                        continue
                    if len(interval_membership[i])==0:
                        check_interval[i] = False
                        continue
                    if ad_test(interval_membership[i])>ad_threshold:
                        split_index.append(i)
                        ad_scores.append(ad_test(interval_membership[i]))
                        modified = True
                    else:
                        check_interval[i] = False
                    
                if not modified:
                    print(f'\tLOG: Convergence after {iteration} iterations.')
                    return cover  
                
                ad_scores = [0 if x != x else x for x in ad_scores]
                if max(ad_scores)==0:
                    print(f'\tLOG: Convergence after {iteration} iterations.')
                    return cover  
                
                best_split=ad_scores.index(max(ad_scores))
                j=split_index[best_split]
                check_interval[j] = True
                check_interval.insert(j+1, True)
                start_time = time.time() 
                split(interval_membership,cover,g_overlap,j)
            
                del ad_scores[best_split]
                del split_index[best_split]
            else:
                best_split=ad_scores.index(max(ad_scores))
                j=split_index[best_split]
                check_interval[j] = True
                check_interval.insert(j+1, True)
                split(interval_membership,cover,g_overlap,j)
               
                del ad_scores[best_split]
                del split_index[best_split]
          
        elif method == 'randomized':
            all_elements_idx = [i for i in range(len(cover.intervals))]
            element_ad_scores = [ad_test(interval_membership[i])
                               for i in range(len(cover.intervals))]
            element_ad_scores = [0 if x != x else x for x in element_ad_scores]
            found_valid = False
            while not found_valid and len(all_elements_idx) != 0:
                # Sample one of the remaining intervals weighted by ad score
                weights = np.asarray(element_ad_scores)[all_elements_idx]
                weights = weights / weights.sum()
                current_element = int(np.random.choice(np.asarray(all_elements_idx), p=weights))
                j=current_element
                if len(interval_membership[j])==0:
                    removal_idx = all_elements_idx.index(j)
                    all_elements_idx.pop(removal_idx)
                    continue
                if ad_test(interval_membership[j])>ad_threshold:
                    check_interval[j] = True
                    check_interval.insert(j+1, True)
                    split(interval_membership,cover,g_overlap,j)
                    found_valid = True
                else:
                    removal_idx = all_elements_idx.index(j)
                    all_elements_idx.pop(removal_idx)
            
            if not found_valid:
                print(f'\tLOG: Convergence after {iteration} iterations.')      
                return cover
            
        elif method == 'DFS':
            for i in range(len(cover.intervals)):
                if not check_interval[i]:
                    continue
                if len(interval_membership[i])==0:
                    check_interval[i] = False
                    continue
                if ad_test(interval_membership[i])>ad_threshold:
                    check_interval[i] = True
                    check_interval.insert(i+1, True)
                    tem=len(interval_membership[i])
                    split(interval_membership,cover,g_overlap,i) 
                    if tem==len(interval_membership[i]):
                        check_interval[i] = False
                        continue
                    if tem==len(interval_membership[i+1]):
                        check_interval[i+1] = False
                        continue
                    ad_scores = [ad_test(interval_membership[i]), ad_test(interval_membership[i+1])]
                    if ad_scores[1]>ad_scores[0]:
                        temp=cover.intervals[i+1]
                        cover.intervals=np.delete(cover.intervals,i+1,axis=0)
                        cover.intervals=np.insert(cover.intervals, i, temp, axis=0)
                        temp=interval_membership[i+1]
                        interval_membership.pop(i+1)
                        interval_membership.insert(i,temp) 
                    modified = True
                    break
                else:
                    check_interval[i] = False
            
            if not modified:
                print(f'\tLOG: Convergence after {iteration} iterations.')
                return cover    
        
        if len(cover.intervals) > max_intervals:
                break
    return cover 