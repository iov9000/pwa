import tensorflow as tf
import numpy as np

def contrast_norm(pics):
    # pics is a [N, H, W, C] tensor
    mean, var = tf.nn.moments(pics, axes=[1, 2, 3], keep_dims=True)
    return pics / tf.sqrt(var + 1e-08)

def log_likelihood(model, optimizer, n=10):
    """
    :param model: model object
    :param optimizer: optimizer object
    :param n: number of MC samples
    :return: MC estimate of log-likelihood
    """

    z = model.q_z.sample(n)

    log_p_z = optimizer.p_z.log_prob(z)

    if model.distribution == 'normal':
        log_p_z = tf.reduce_sum(log_p_z, axis=-1)

    log_p_x_z = -tf.reduce_sum(optimizer.bce, axis=-1)

    log_q_z_x = model.q_z.log_prob(z)

    if model.distribution == 'normal':
        log_q_z_x = tf.reduce_sum(log_q_z_x, axis=-1)

    return tf.reduce_mean(tf.reduce_logsumexp(
        tf.transpose(log_p_x_z + log_p_z - log_q_z_x) - np.log(n), axis=-1))

def sq_distances(x,y,scaled=True):
    sq_norms_x = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
    sq_norms_y = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
    dotprods = tf.matmul(x, y, transpose_b=True)
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*||x||*||y||
    d = sq_norms_x + tf.transpose(sq_norms_y) - 2. * dotprods
    d_scaled = d/((1-sq_norms_x)*(1-sq_norms_y))
    d_scaled = 1+2*tf.where(d_scaled<1.0, tf.ones_like(d_scaled)+1e-10, d_scaled)
    return sq_norms_x, sq_norms_y, d_scaled

def hyp_distances(x,y,q,c=1.0):
    sq_norms_x, sq_norms_y, d_xy = sq_distances(x,y)
    #dotprods = tf.matmul(points, points, transpose_b=True)
    #sq_dist = sq_norms + tf.transpose(sq_norms) - 2. * dotprods
    hyp_dist = tf.acosh(d_xy+1e-6)**q
    return sq_norms_x, d_xy, hyp_dist

def block_diagonal(matrices, dtype=tf.float32):
    matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
    blocked_rows = tf.Dimension(0)
    blocked_cols = tf.Dimension(0)
    batch_shape = tf.TensorShape(None)
    for matrix in matrices:
        full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
        batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
        blocked_rows += full_matrix_shape[-2]
        blocked_cols += full_matrix_shape[-1]
    ret_columns_list = []
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        ret_columns_list.append(matrix_shape[-1])
    ret_columns = tf.add_n(ret_columns_list)
    row_blocks = []
    current_column = 0
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        row_before_length = current_column
        current_column += matrix_shape[-1]
        row_after_length = ret_columns - current_column
        row_blocks.append(tf.pad(
            tensor=matrix,
            paddings=tf.concat(
                [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
                [(row_before_length, row_after_length)]],
                axis=0)))
    blocked = tf.concat(row_blocks, -2)
    blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
    return blocked

def mmd_penalty(params, sample_pz, sample_qhat, sigma=None, q=1.0):
    # assert opts["e_noise"] in ('gaussian'), \
        # 'MMD++ works only with Gaussian encoders!'
    # Number of codes per the same pic
    ns = params['nr_mmd_samples']
    n = params['batch_size']
    ls = params['embedding_dim']
    kernel = params['mmd_kernel']
    sigma2_p = 1 # params['pz_scale'] ** 2
    params['pz'] = 'normal'

    sq_norms_pz, _, dist_pz = hyp_distances(sample_pz,sample_pz,q,params['c'])
    sq_norms_qhat, dist_qhat_prime, dist_qhat = hyp_distances(sample_qhat,sample_qhat,q,params['c'])
    _, _, dist_pz_qhat = hyp_distances(sample_pz,sample_qhat,q,params['c'])

    mask = block_diagonal(
      [np.ones((ns, ns), dtype=np.float32) for i in range(n)],
      tf.float32)

    # Median heuristic for the sigma^2 of Gaussian kernel
    l = int(n/2)
    if sigma is None:
        sigma2_k = tf.nn.top_k(
          tf.reshape(dist_pz_qhat, [-1]), l).values[l - 1]
        sigma2_k += tf.nn.top_k(
          tf.reshape(dist_qhat, [-1]), l).values[l - 1]
    else:
        sigma2_k = sigma

    #if params['verbose']:
    #    sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
    # Part (1)
    res1 = tf.exp( - dist_pz / q / sigma2_k)
    res1 = tf.multiply(res1, 1. - tf.eye(n)) # filter i=j pairs of samples
    res1 = tf.reduce_sum(res1) / (n * n - n)
    # Part (2)
    res2 = tf.exp( - dist_pz_qhat / q / sigma2_k)
    res2 = tf.reduce_sum(res2) / (n * n) / ns
    # Part (3)
    res3 = tf.exp( - dist_qhat / q / sigma2_k)
    res3 = tf.multiply(res3, 1. - tf.eye(n)) # filter i=j pairs of samples
    res3 = tf.reduce_sum(res3) / (n * n - n)
    stat = res1 - 2 * res2 + res3 

    return stat 

def kld_loss(mu,sigma,d):
    Z = tf.reduce_sum(2*np.pi*tf.sqrt(np.pi/2)*sigma*tf.exp(sigma**2/2)*tf.erf(sigma/tf.sqrt(2.0)),axis=1)
    mu_norm = tf.norm(mu,2,axis=1)
    return tf.reduce_mean(mu_norm**2-mu_norm*(np.sqrt(2.0)*tf.lgamma((d+1)/2)/tf.lgamma(d/2))-tf.log(Z))

def mc_elbo(x_hat, z, mu, log_sigma):
    dist = tf.contrib.distributions.MultivariateNormalDiag(x_hat, tf.ones_like(x_hat))
    log_p_x_z = dist.log_prob(x_hat)
    mean = tf.constant(np.array([[0.0,0.0]]),tf.float32)
    log_p_z = gauss_prob_pd(z,mean,1)
    log_q_z_x = gauss_prob_pd(z,mu,tf.exp(0.5*log_sigma))
    return log_p_x_z + log_p_z - log_q_z_x