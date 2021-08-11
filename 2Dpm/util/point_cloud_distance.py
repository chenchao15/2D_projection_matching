# MIT License
#
# Copyright (c) 2018 Chen-Hsuan Lin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import numpy as np
    

def point_cloud_distance(Vs, Vt):
    """
    For each point in Vs computes distance to the closest point in Vt, only for test.
    """
    VsN = tf.shape(Vs)[0]
    VtN = tf.shape(Vt)[0]
    Vt_rep = tf.tile(Vt[None, :, :], [VsN, 1, 1])  # [VsN,VtN,3]
    Vs_rep = tf.tile(Vs[:, None, :], [1, VtN, 1])  # [VsN,VtN,3]
    diff = Vt_rep-Vs_rep
    dist = tf.sqrt(tf.reduce_sum(diff**2, axis=[2]))  # [VsN,VtN]
    idx = tf.to_int32(tf.argmin(dist, axis=1))
    proj = tf.gather_nd(Vt_rep, tf.stack([tf.range(VsN), idx], axis=1))
    minDist = tf.gather_nd(dist, tf.stack([tf.range(VsN), idx], axis=1))
    return proj, minDist, idx


def another_chamfer_distance(Vs,Vt):
    Vs = Vs[0]
    Vt = Vt[0]
    VsN = tf.shape(Vs)[0] 
    VtN = tf.shape(Vt)[0]
    Vt_rep = tf.tile(Vt[None, :, :], [VsN, 1, 1])  # [VsN,VtN,3]
    Vs_rep = tf.tile(Vs[:, None, :], [1, VtN, 1])  # [VsN,VtN,3]
    diff = Vt_rep-Vs_rep
    dist = tf.sqrt(tf.reduce_sum(diff**2, axis=[2]))  # [VsN,VtN]
    idx = tf.to_int32(tf.argmin(dist, axis=1))
    n = tf.stack([tf.range(VsN), idx], axis=1)
    minDist = tf.gather_nd(dist, tf.stack([tf.range(VsN), idx], axis=1))
    return minDist, idx


def chamfer_distance(Vs, Vt):
    """
    For each point in Vs computes distance to the closest point in Vt
    """
    batch_size = tf.shape(Vs)[0]
    VsN = tf.shape(Vs)[1]
    VtN = tf.shape(Vt)[1]
    Vt_rep = tf.tile(Vt[:,None, :, :], [1,VsN, 1, 1])  # [VsN,VtN,3]
    Vs_rep = tf.tile(Vs[:,:, None, :], [1,1, VtN, 1])  # [VsN,VtN,3] (1024, ?, 2)
    diff = Vt_rep-Vs_rep
    dist = tf.sqrt(tf.reduce_sum(diff**2, axis=[3]))  # [VsN,VtN] (1024, 1024)
    minDist = tf.reduce_min(dist, 2)
    return minDist

	
def euclidean_distance_self(V):
    VN = tf.shape(V)[1]
    V_rep = V[:,:,None,:]
    V_rep_T = tf.transpose(V_rep, [0,2,1,3])
    diff = V_rep - V_rep_T
    dist = tf.sqrt(tf.reduce_sum(diff**2, axis=[3]) + 1e-8)
    return dist


def kl_distance_basic(p, q):
    plength = p.shape[1]
    scale = int(2000/200)
    for i in range(plength): 
        if not i % scale == 0:  
            continue 
        p_vec = p[:,i,:]
        p_vec = p_vec[:,None,:]
        q_vec = q
        kl_pq = tf.reduce_sum(p_vec * (tf.log(p_vec) - tf.log(q_vec)), 2)
        if i == 0:
            res = tf.expand_dims(kl_pq, 1)
        else:
            kl_pq = tf.expand_dims(kl_pq, 1)
            res = tf.concat([res, kl_pq], 1) 
    return res


def kl_distance(Vs, Vt):
    def kl(V):
        VN = tf.shape(V)[1]
        V_rep = V[:,:,None,:]
        V_rep_T = tf.transpose(V_rep, [0,2,1,3])
        diff = V_rep - V_rep_T
        dist = tf.reduce_sum(diff**2, axis=[3])
        exp_dist = tf.exp(-dist)
        temp_exp_dist_sum = tf.expand_dims(tf.reduce_sum(exp_dist, 2), 2)
        exp_dist_sum = tf.tile(temp_exp_dist_sum, [1, 1, VN])
        pl = tf.divide(exp_dist, exp_dist_sum)
        return pl, exp_dist, dist, exp_dist_sum
        
    p,distp,_,_ = kl(Vs)
    q,dist,dis,exp_sum = kl(Vt)
    vp = p.shape[1]
    vq = q.shape[1]
    KL = kl_distance_basic(p, q)
    dis = tf.reduce_min(KL, 2)
    return tf.reduce_mean(dis)


def kl_distance_topk(Vs, Vt):
    def kl(V):
        VN = tf.shape(V)[1]
        V_rep = V[:,:,None,:]
        V_rep_T = tf.transpose(V_rep, [0,2,1,3])
        diff = V_rep - V_rep_T
        dist = tf.reduce_sum(diff**2, axis=[3])
        exp_dist = tf.exp(-dist)
        max_dist = tf.reduce_max(exp_dist) - exp_dist
        dist_top_max = tf.nn.top_k(max_dist, k=10)
        mindist = tf.reduce_max(exp_dist) - dist_top_max.values
        temp_exp_dist_sum = tf.expand_dims(tf.reduce_sum(mindist, 2), 2) 
        exp_dist_sum = tf.tile(temp_exp_dist_sum, [1,1,10])
        pl = tf.divide(mindist, exp_dist_sum)
        return pl

    p = kl(Vs)
    q = kl(Vt)
    KL = kl_distance_basic(p, q)
    KL_T = tf.transpose(KL, [0,2,1])
    dis1 = tf.reduce_min(KL, 2)
    dis2 = tf.reduce_min(KL_T, 2)  
    return dis1 + dis2


def euclidean_distance_for_two_points(Vs, Vt):
    batch_size = tf.shape(Vs)[0]
    VsN = tf.shape(Vs)[1] 
    VtN = tf.shape(Vt)[1] 
    Vt_rep = tf.tile(Vt[:,None, :, :], [1,VsN, 1, 1])  # [VsN,VtN,3]    
    Vs_rep = tf.tile(Vs[:,:, None, :], [1,1, VtN, 1])  # [VsN,VtN,3] (1024, ?, 2) 
    diff = Vt_rep-Vs_rep
    mm = tf.square(diff)
    dist = mm[:,:,:,0] + mm[:,:,:,1] # [VsN,VtN]
    return dist


def euclidean_distance_for_fuzz_pc(Vs,Vt):
    batch_size = tf.shape(Vs)[0]
    VsN = tf.shape(Vs)[1]
    VtN = tf.shape(Vt)[1] 
    num = tf.shape(Vs)[2]
    
    for j in range(1):
        Vt_rep = tf.tile(Vt[:,None,:,j,:], [1,VsN,1,1])
        Vs_rep = tf.tile(Vs[:,:,None,j,:], [1,1,VtN,1])
        diff = tf.square(Vt_rep - Vs_rep)
        dist = tf.sqrt(diff[:,:,:,0]+diff[:,:,:,1]+diff[:,:,:,2]+1e-16)
        if j == 0:
            res = dist[:,:,:,None]
        else:
            res = tf.expand_dims(tf.reduce_min(tf.concat([res, dist[:,:,:,None]], 3), 3), 3)
    
    return res


def another_euclidean_distance_for_fuzz_pc(Vs, Vt):
    batch_size = tf.shape(Vs)[0]
    VsN = tf.shape(Vs)[1]
    VtN = tf.shape(Vt)[1]
    Vt_rep = 128 * tf.tile(Vt[:,None,:,:,:], [1,VsN,1,1,1])
    Vs_rep = 128 * tf.tile(Vs[:,:,None,:,:], [1,1,VtN,1,1])
    diff = Vt_rep - Vs_rep
    diff = tf.abs(diff)
    diff = tf.cast(diff, 'float16')
    a = diff[:,:,:,:,0] 
    b = diff[:,:,:,:,1]
    c = diff[:,:,:,:,2]
    dist = a + b + c
    return dist, Vs_rep


def chamfer_distance_topk(Vs, Vt, sumk):
    batch_size = tf.shape(Vs)[0]
    VsN = tf.shape(Vs)[1]
    VtN = tf.shape(Vt)[1]
    Vt_rep = tf.tile(Vt[:,None, :, :], [1,VsN, 1, 1])  # [VsN,VtN,3] 
    Vs_rep = tf.tile(Vs[:,:, None, :], [1,1, VtN, 1])  # [VsN,VtN,3] (1024, ?, 2)
    diff = Vt_rep-Vs_rep
    dist = tf.reduce_sum(diff**2, axis=[3])  # [VsN,VtN] (1024, 1024)
    maxDist = tf.nn.top_k(dist, k=sumk)
    maxDist = maxDist.values
    dis_dist = tf.reduce_max(dist) - dist
    minDist = tf.nn.top_k(dis_dist, k=sumk)
    minDist = tf.reduce_max(dist) - minDist.values
    return minDist

 
def chamfer_distance3D(Vs, Vt):
    batch_size = tf.shape(Vs)[0]
    VsN = tf.shape(Vs)[1]
    VtN = tf.shape(Vt)[1]
    Vt_rep = tf.tile(Vt[:,None, :, :], [1,VsN, 1, 1])  # [batch_size,VsN,VtN,3]
    Vs_rep = tf.tile(Vs[:,:, None, :], [1,1, VtN, 1])  # [batch_size, VsN,VtN,3] (5,2000, ?, 3)
    diff = Vt_rep-Vs_rep
    dist = tf.sqrt(tf.reduce_sum(diff**2, axis=[3]))  # [batch_size,VsN,VtN] (5,2000,?)
    dist_liner = tf.reshape(dist, [batch_size * VsN, VtN])  #[batch_size * VsN,VtN]
    idx = tf.to_int32(tf.argmin(dist_liner, axis=1))
    minDist_liner = tf.gather_nd(dist_liner, tf.stack([tf.range(batch_size * VsN), idx], axis=1))
    minDist = tf.reshape(minDist_liner,[batch_size, VsN])
    return minDist, idx


def chamfer_distance_self(Vs, max_size=128.0):
    batch_size, n_points, _ = Vs.shape
    row = tf.tile(Vs, [1,n_points,1])
    Vs = tf.expand_dims(Vs, [2])
    line = tf.reshape(tf.tile(Vs, [1,1,n_points, 1]), [batch_size, n_points * n_points, 2])
    distance_liner = tf.reduce_sum(tf.square(tf.subtract(row, line)), 2)    
    distance = tf.reshape(distance_liner, [batch_size, n_points, n_points])
    diag_list = tf.cast(max_size, 'float32') * tf.ones([batch_size, n_points], dtype=tf.float32)
    distance = distance + tf.matrix_diag(diag_list)
    chamfer_distance = tf.reduce_min(distance, 1)
    return chamfer_distance, distance

