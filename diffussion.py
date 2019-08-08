# Author: Dmytro Mishkin, 2018.
# This is simple partial re-implementation of papers 
# Fast Spectral Ranking for Similarity Search, CVPR 2018. https://arxiv.org/abs/1703.06935
# and  Efficient Diffusion on Region Manifolds: Recovering Small Objects with Compact CNN Representations, CVPR 2017 http://people.rennes.inria.fr/Ahmet.Iscen/diffusion.html

import time
import os
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix, eye, diags
from scipy.sparse import linalg as s_linalg

def sim_kernel(dot_product):
    return np.maximum(np.power(dot_product,3),0)
 
def normalize_connection_graph(G):
    start_time = time.time()
    W = csr_matrix(G)
    W = W - diags(W.diagonal())
    D = np.array(1./ np.sqrt(W.sum(axis = 1)))
    D[np.isnan(D)] = 0
    D[np.isinf(D)] = 0
    D_mh = diags(D.reshape(-1))
    Wn = D_mh * W * D_mh
    duration = time.time() - start_time
    minute = int(duration / 60)
    second = int(duration) % 60
    print("%dminutes %dseconds" % (minute,second)) 
    return Wn

def topK_W(G, K = 100):
    sortidxs = np.argsort(-G, axis = 1)
    for i in range(G.shape[0]):
        G[i,sortidxs[i,K:]] = 0
    G = np.minimum(G, G.T)
    return G
def find_trunc_graph(qs, W, levels = 3):
    needed_idxs = []
    needed_idxs = list(np.nonzero(qs > 0)[0])
    for l in range(levels):
        idid = W.nonzero()[1]
        needed_idxs.extend(list(idid))
        needed_idxs =list(set(needed_idxs))
    return np.array(needed_idxs), W[needed_idxs,:][:,needed_idxs]

def dfs_trunk(sim, A,alpha = 0.99, QUERYKNN = 10, maxiter = 8, K = 100, tol = 1e-3):
    start_time = time.time()
    qsim = sim_kernel(sim).T
    sortidxs = np.argsort(-qsim, axis = 1)
    for i in range(len(qsim)):
        qsim[i,sortidxs[i,QUERYKNN:]] = 0
    qsims = sim_kernel(qsim)
    W = sim_kernel(A)
    W = csr_matrix(topK_W(W, K))
    out_ranks = []
    t =time()
    for i in range(qsims.shape[0]):
        qs =  qsims[i,:]
        tt = time() 
        w_idxs, W_trunk = find_trunc_graph(qs, W, 2);
        Wn = normalize_connection_graph(W_trunk)
        Wnn = eye(Wn.shape[0]) - alpha * Wn
        f,inf = s_linalg.minres(Wnn, qs[w_idxs], tol=tol, maxiter=maxiter)
        ranks = w_idxs[np.argsort(-f.reshape(-1))]
        missing = np.setdiff1d(np.arange(A.shape[1]), ranks)
        out_ranks.append(np.concatenate([ranks.reshape(-1,1), missing.reshape(-1,1)], axis = 0))

    out_ranks = np.concatenate(out_ranks, axis = 1)
    duration = time.time() - start_time
    minute = int(duration / 60)
    second = int(duration) % 60
    print("%dminutes %dseconds" % (minute,second))   
    return out_ranks

def cg_diffusion(qsims, Wn, alpha = 0.99, maxiter = 10, tol = 1e-3):
    Wnn = eye(Wn.shape[0]) - alpha * Wn
    out_sims = []
    for i in range(qsims.shape[0]):
        #f,inf = s_linalg.cg(Wnn, qsims[i,:], tol=tol, maxiter=maxiter)
        f,inf = s_linalg.minres(Wnn, qsims[i,:], tol=tol, maxiter=maxiter)
        out_sims.append(f.reshape(-1,1))
    out_sims = np.concatenate(out_sims, axis = 1)
    ranks = np.argsort(-out_sims, axis = 0)
    return ranks

def fsr_rankR(qsims, Wn, alpha = 0.99, R = 2000):
    start_time = time.time()
    vals, vecs = s_linalg.eigsh(Wn, k = R)
    p2 = diags((1.0 - alpha) / (1.0 - alpha*vals))
    vc = csr_matrix(vecs)
    p3 =  vc.dot(p2)
    ##vc_norm =  (vc.multiply(vc)).sum(axis = 0)
    out_sims = []
    for i in range(qsims.shape[0]):
        qsims_sparse = csr_matrix(qsims[i:i+1,:])
        p1 =(vc.T).dot(qsims_sparse.T)
        diff_sim = csr_matrix(p3)*csr_matrix(p1)
        out_sims.append(diff_sim.todense().reshape(-1,1))
    out_sims = np.concatenate(out_sims, axis = 1)
    ranks = np.argsort(-out_sims, axis = 0)
    duration = time.time() - start_time
    minute = int(duration / 60)
    second = int(duration) % 60
    print("%dminutes %dseconds" % (minute,second))   
    return ranks.T



def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float16)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int16)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float16)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float16)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float16)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist