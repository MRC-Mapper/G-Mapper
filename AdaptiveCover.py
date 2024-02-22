import numpy as np
import numpy.linalg as LA
from typing import List, Dict
from math import log2, log, pi

from cover import Cover
from mapper import generate_mapper_graph
import copy

def bic_centroid(X, c, assignments, BIC=True):
    '''
    Method for computing the BIC or AIC. 
    @param: X - the point cloud to evaluate over. n points x d dimensions
    @param: c - cluster centroids. c centroids x d dimensions
    @params: assigments - Cluster assignments for n points.
    @params: BIC - Defaults to True. Setting False computes the AIC.
    '''
    k = len(c)
    d = X.shape[1]
    R = X.shape[0]
    var = 0
    log_term = 0

    # Sometimes, in sparse cases, a node gets completely split and all members join the neighbors
    empty_clusters = []
    set_assignments = set(assignments)
    for i in range(k):
        if i not in set_assignments:
            empty_clusters.append(i)

    assignments = np.asarray(assignments, dtype=np.int)
    for i in range(k):
        if i in empty_clusters:
            continue
        cluster_members = X[assignments == i]
        log_term += cluster_members.shape[0] * \
            (log(cluster_members.shape[0]) - log(R))
        sum_squares = np.linalg.norm(cluster_members - c[i], axis=1) ** 2
        var = var + sum_squares.sum()
    k = k - len(empty_clusters)
    var = var / (R - k)
    t2 = -1 * (R*d / 2) * log(2*pi*var)
    t3 = -1 * (1 / 2) * (R-k)
    llh = log_term + t2 + t3
    if BIC:
        return llh - ((k * (d+1))/2) * log(R)  # bic
    else:
        return 2 * llh - ((k * (d + 1))) * 2  # aic


def BIC_Cover_Centroid(X, lens, perc_overlap, min_intervals, max_intervals, interval_step, clusterer, BIC=True):
    '''
    Runs an exhaust search over [min_intervals, max_intervals] in intervals_step steps. Returns the BIC or AIC 
    cost for each interval parameter.
    '''
    # Returns optimal cover object, costs, num_clusters
    costs = []
    intervals = [i for i in range(min_intervals, max_intervals, interval_step)]

    for interval in intervals:
        current_cover = Cover(num_intervals=interval,
                              percent_overlap=perc_overlap, enhanced=False)
        graph = generate_mapper_graph(X, lens, current_cover, clusterer)
        centroids, membership, _ = graph.to_hard_clustering_set(X)
        costs.append(bic_centroid(X, centroids, membership, BIC))
    return costs, intervals


def adaptive_cover_BFS(X, lens, initial_cover, clusterer, iterations, max_intervals, BIC, delta, debug):
    '''
    Runs the BFS version of xmeans-adaptive-cover. See xmeans_adaptive_cover for parameters and defaults.
    The BFS version checks each interval in order, splitting the best interval after examining all.
    '''
    cover = initial_cover
    # None for not computed - caches since modifying one interval does not change assignments for other intervals.
    BIC_difference_cache = [None for i in range(cover.num_intervals)] 
    BIC_before_cache = [None for i in range(cover.num_intervals)]
    BIC_after_cache = [None for i in range(cover.num_intervals)]

    for iteration in range(iterations):
        if debug is not None:
            print('iteration', iteration)
        # Generate the first mapper graph to compare each new generated one
        g = generate_mapper_graph(
            X, lens, cover, clusterer, enhanced=False, refit_cover=False)

        for i in range(cover.num_intervals):
            if BIC_difference_cache[i] is not None:
                continue
            # Compute the bic presplit
            interval_centers, interval_membership, interval_members = g.to_hard_clustering_set(
                X, intervals=[i])
            # Case the interval doesn't have any clusters
            if len(interval_centers) == 0:
                BIC_difference_cache[i] = -10
                continue
            old_bic = bic_centroid(
                X[interval_members], interval_centers, interval_membership, BIC=BIC)

            # Generate the split
            cover.divide_interval(i)
            g_split = generate_mapper_graph(
                X, lens, cover, clusterer, enhanced=False, refit_cover=False)
            interval_centers, interval_membership, interval_members = g_split.to_hard_clustering_set(
                X, intervals=[i, i+1])
            # Case where no clusters are formed post-split
            if len(interval_centers) == 0:
                cover.merge_interval(i, i+1)
                BIC_difference_cache[i] = -10
                continue
            
            # Store values
            new_bic = bic_centroid(
                X[interval_members], interval_centers, interval_membership, BIC=BIC)
            BIC_difference_cache[i] = new_bic - old_bic
            BIC_before_cache[i] = old_bic
            BIC_after_cache[i] = new_bic
            cover.merge_interval(i, i+1)

        # Sort the differences so we check the largest difference intervals first
        BIC_difference_idx = np.argsort(np.asarray(BIC_difference_cache))
        modified = False
        num_elements = BIC_difference_idx.shape[0]
        for i in range(1, num_elements+1):
            if modified:
                continue
            idx = int(BIC_difference_idx[num_elements-i])
            bic_before = BIC_before_cache[idx]
            if bic_before is None:
                continue
            if BIC_difference_cache[idx] >= delta * bic_before:
                # If a valid split is found, keep it and update book-keeping
                modified = True
                BIC_difference_cache[idx] = None
                BIC_difference_cache.insert(idx, None)
                BIC_before_cache[idx] = None
                BIC_before_cache.insert(idx, None)
                BIC_after_cache[idx] = None
                BIC_after_cache.insert(idx, None)
                cover.divide_interval(idx)
        
        # Termination conditions and debugging output
        if not modified:
            print(f'\tLOG: Convergence after {iteration} iterations.')
            cover.remove_duplicate_cover_elements()
            return cover
        if debug is not None:
            cover.vis_last_split(iteration=iteration, loc=debug)
        if cover.num_intervals > max_intervals:
            break

    cover.remove_duplicate_cover_elements()
    return cover

def adaptive_cover_randomized(X, lens, initial_cover, clusterer, iterations, max_intervals, BIC, delta, debug):
    '''
    Runs the randomized version of xmeans-adaptive-cover. See xmeans_adaptive_cover for parameters and defaults.
    The randomized version checks the interval randomly weighted by the length of the interval (biases towards unexplored regions).
    '''
    cover = initial_cover
    num_iter = 0
    # If we check an interval and realize it can't be split, we can skip subsequent checks.
    skip_interval = []
    for iteration in range(iterations):
        if debug is not None:
            print('iteration', iteration)
        num_iter = iteration
        # Generate the current mapper graph
        g = generate_mapper_graph(
            X, lens, cover, clusterer, enhanced=False, refit_cover=False)
        
        # Generate probability weighting
        all_elements_idx = [i for i in range(cover.num_intervals)]
        element_lengths = [cover[i][1] - cover[i][0]
                            for i in range(cover.num_intervals)]
        found_valid = False # Denotes finding a valid split
        while not found_valid and len(all_elements_idx) != 0:
            # Sample one of the remaining intervals weighted by length
            weights = np.asarray(element_lengths)[all_elements_idx]
            weights = weights / weights.sum()
            current_element = int(np.random.choice(
                np.asarray(all_elements_idx), p=weights))

            # Already have checked this interval and deemed it can't be split            
            if current_element in skip_interval:
                removal_idx = all_elements_idx.index(current_element)
                all_elements_idx.pop(removal_idx)
                continue

            # Compute the old BIC
            interval_centers, interval_membership, interval_members = g.to_hard_clustering_set(
                X, intervals=[current_element])
            # Case there are no clusters
            if len(interval_centers) == 0:
                removal_idx = all_elements_idx.index(current_element)
                all_elements_idx.pop(removal_idx)
                continue
            old_bic = bic_centroid(
                X[interval_members], interval_centers, interval_membership, BIC=BIC)

            # Generate the split and new mapper graph
            cover.divide_interval(current_element)
            g_split = generate_mapper_graph(
                X, lens, cover, clusterer, enhanced=False, refit_cover=False)

            # Compute the new BIC score
            interval_centers, interval_membership, interval_members = g_split.to_hard_clustering_set(
                X, intervals=[current_element, current_element + 1])
            # Case post-split there are no clusters
            if len(interval_centers) == 0:
                cover.merge_interval(current_element, current_element+1)
                removal_idx = all_elements_idx.index(current_element)
                all_elements_idx.pop(removal_idx)
                continue
            new_bic = bic_centroid(
                X[interval_members], interval_centers, interval_membership, BIC=BIC)

            # If the split is valid, keep it
            if (new_bic - old_bic) >= delta * old_bic:
                found_valid = True
                new_skip = []
                for elem in skip_interval:
                    # Reindexing of the intervals we don't need to check
                    if elem < current_element:
                        new_skip.append(elem)
                    else:
                        new_skip.append(elem + 1)
            else: # Can't split this interval. Merge the split back up and book-keeping
                cover.merge_interval(current_element, current_element + 1)
                removal_idx = all_elements_idx.index(current_element)
                all_elements_idx.pop(removal_idx)

        # If no changes are made, exit prematurely after removing duplicates
        if not found_valid:
            print(f'\tLOG: Convergence after {num_iter} iterations.')
            cover.remove_duplicate_cover_elements()
            return cover
        
        if debug is not None:
            cover.vis_last_split(iteration=iteration, loc=debug)
       
        if cover.num_intervals > max_intervals:
            break

    cover.remove_duplicate_cover_elements()
    return cover

def adaptive_cover_DFS(X, lens, initial_cover, clusterer, iterations, max_intervals, BIC, delta, debug):    
    '''
    Runs the DFS version of xmeans-adaptive-cover. See xmeans_adaptive_cover for parameters and defaults.
    The DFS version checks each interval in order. As soon as an interval can be split, it splits it. This explores as deep as possible before continuing.
    '''
    cover=copy.deepcopy(initial_cover)
    if (str(type(cover.intervals))=="<class 'NoneType'>"):
        cover.compute_intervals(np.min(lens), np.max(lens), lens)
    num_iter = 0
    

    # Default every interval should be checked
    check_interval = [True for i in range(cover.num_intervals)]
    for iteration in range(iterations):
        if debug is not None:
            print('iteration', iteration)
        num_iter = iteration
        modified = False
        g = generate_mapper_graph(
            X, lens, cover, clusterer, enhanced=False, refit_cover=False)
        
        for i in range(cover.num_intervals):
            if not check_interval[i]:
                continue
            # Compute the bic presplit
            interval_centers, interval_membership, interval_members = g.to_hard_clustering_set(
                X, intervals=[i])
            # If there are no clusters in the interval
            if len(interval_centers) == 0:
                continue
            old_bic = bic_centroid(
                X[interval_members], interval_centers, interval_membership, BIC=BIC)

            # Generate the split
            cover.divide_interval(i)
            g_split = generate_mapper_graph(
                X, lens, cover, clusterer, enhanced=False, refit_cover=False)
            interval_centers, interval_membership, interval_members = g_split.to_hard_clustering_set(
                X, intervals=[i, i+1])
            # Case there are no clusters
            if len(interval_centers) == 0:
                cover.merge_interval(i, i+1)
                continue

            new_bic = bic_centroid(
                X[interval_members], interval_centers, interval_membership, BIC=BIC)
            if (new_bic - old_bic) >= delta * old_bic:
                # we have a good candidate and keep it
                modified = True
                check_interval.insert(i, True)
                break
            else:
                # We know this interval can't be split to remember that to save compute
                check_interval[i] = False
                cover.merge_interval(i, i+1)

        if not modified:
            print(f'\tLOG: Convergence after {num_iter} iterations.')
            cover.remove_duplicate_cover_elements()
            return cover

        if debug is not None:
            cover.vis_last_split(iteration=iteration, loc=debug)

        if cover.num_intervals > max_intervals:
            break

    cover.remove_duplicate_cover_elements()
    return cover


def xmeans_adaptive_cover(X, lens, initial_cover, clusterer, iterations=10, max_intervals=10, BIC=True, delta=0., method='BFS', debug=None): 
    '''
    xmeans adaptive mapper cover as outlined in our paper. The main driver method. You should use this to access the other methods.
    @param: X - n points x d dimensions point cloud np.ndarray
    @param: lens - n dimensional vector for the lens (only 1D mapper is supported)
    @param: initial_cover - Starting cover. Must be an cover.AbstractCover object
    @param: clusterer - Sklearn clusterer. E.g. DBSCAN. Must follow the fit_predict api.
    @param: iterations - max number of iterations to run.
    @param: max_intervals - maximum number of intervals
    @param: BIC - True indicates using BIC and False indicates AIC
    @param: delta - Delta in the paper. Percent difference considered as "convergent".
            New BIC score >= delta * old BIC score to keep a split. Specified as a fraction.
    @param: method: One of 'BFS', 'DFS', or 'randomized'. Defaults to 'BFS'.
    @param: debug: None or location of a directory to save intermediate representations of the splitting steps.
    '''
    if method == 'BFS' or method is None:
        return adaptive_cover_BFS(X, lens, initial_cover, clusterer, iterations, max_intervals, BIC, delta, debug)
    if method == 'DFS':
        return adaptive_cover_DFS(X, lens, initial_cover, clusterer, iterations, max_intervals, BIC, delta, debug)
    if method == 'randomized' or method is None:
        return adaptive_cover_randomized(X, lens, initial_cover, clusterer, iterations, max_intervals, BIC, delta, debug)


  