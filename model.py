import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import RMSprop
from data_fetcher import data_loader
import config

import multiprocessing

cores = multiprocessing.cpu_count()

tf.random.set_seed(3615)
np.random.seed(3615)
tfd = tfp.distributions

dims = config.emb_dim
batch_size = 128
regularization = 100
dl = data_loader(batch_size, 'item')

n_users = dl.n_user
n_items = dl.n_item
print("num users: %d " % n_users)
print("num_items: %d " % n_items)

optimizer = RMSprop(1e-3)
initdelta = 0.05

if config.load_pretrained_emb:
    # item space generator
    i_g_user_emb_matrix = tf.Variable(dl.item_space_user_embedding, name='i_g_user', dtype=tf.float32)
    i_g_item_emb_matrix = tf.Variable(dl.item_space_item_embedding, name='i_g_item', dtype=tf.float32)
    i_g_bias = tf.Variable(tf.zeros([n_items]), dtype=tf.float32, name='i_g_bias')

    # user space generator
    u_g_user_emb_matrix = tf.Variable(dl.social_space_user_embedding, name='u_g_user', dtype=tf.float32)
    u_g_bias = tf.Variable(tf.zeros([n_users]), dtype=tf.float32, name='u_g_bias')

    # item space discriminator
    i_d_user_emb_matrix = tf.Variable(dl.item_space_user_embedding, name='i_d_user', dtype=tf.float32)
    i_d_item_emb_matrix = tf.Variable(dl.item_space_item_embedding, name='i_d_item', dtype=tf.float32)
    i_d_bias = tf.Variable(tf.zeros([n_items]), dtype=tf.float32, name='i_d_bias')

    # user space discriminator
    u_d_user_emb_matrix = tf.Variable(dl.social_space_user_embedding, name='u_d_user', dtype=tf.float32)
    u_d_bias = tf.Variable(tf.zeros([n_users]), dtype=tf.float32, name='u_d_bias')
else:
    # item space generator
    i_g_user_emb_matrix = tf.Variable(tf.random.uniform([n_users, dims], name='i_g_user', minval=-initdelta, maxval=initdelta))
    i_g_item_emb_matrix = tf.Variable(tf.random.uniform([n_items, dims], name='i_g_item', minval=-initdelta, maxval=initdelta))
    i_g_bias = tf.Variable(tf.zeros([n_items]), dtype=tf.float32, name='i_g_bias')

    # user space generator
    u_g_user_emb_matrix = tf.Variable(tf.random.uniform([n_users, dims], name='u_g_user', minval=-initdelta, maxval=initdelta))
    u_g_bias = tf.Variable(tf.zeros([n_users]), dtype=tf.float32, name='u_g_bias')

    # item space discriminator
    i_d_user_emb_matrix = tf.Variable(tf.random.uniform([n_users, dims], name='i_d_user', minval=-initdelta, maxval=initdelta))
    i_d_item_emb_matrix = tf.Variable(tf.random.uniform([n_items, dims], name='i_d_item', minval=-initdelta, maxval=initdelta))
    i_d_bias = tf.Variable(tf.zeros([n_items]), dtype=tf.float32, name='i_d_bias')

    # user space discriminator
    u_d_user_emb_matrix = tf.Variable(tf.random.uniform([n_users, dims], name='u_d_user', minval=-initdelta, maxval=initdelta))
    u_d_bias = tf.Variable(tf.zeros([n_users]), dtype=tf.float32, name='u_d_bias')

# social space to item space mapping
si_w1 = tf.Variable(tf.random.normal([dims, 32], name='si_w1', mean=0, stddev=0.01))
si_b1 = tf.Variable(tf.zeros([32]), name='si_b1')
si_w2 = tf.Variable(tf.random.normal([32, 128], name='si_w2', mean=0, stddev=0.01))
si_b2 = tf.Variable(tf.zeros([128]), name='si_b2')
si_w3 = tf.Variable(tf.random.normal([128, 32], name='si_w3', mean=0, stddev=0.01))
si_b3 = tf.Variable(tf.zeros([32]), name='si_b3')
si_w4 = tf.Variable(tf.random.normal([32, dims], name='si_w4', mean=0, stddev=0.01))
si_b4 = tf.Variable(tf.zeros([dims]), name='si_b4')


# item space to social space mapping
is_w1 = tf.Variable(tf.random.normal([dims, 32], name='is_w1', mean=0, stddev=0.01))
is_b1 = tf.Variable(tf.zeros([32]), name='is_b1')
is_w2 = tf.Variable(tf.random.normal([32, 128], name='is_w2', mean=0, stddev=0.01))
is_b2 = tf.Variable(tf.zeros([128]), name='is_b2')
is_w3 = tf.Variable(tf.random.normal([128, 32], name='is_w3', mean=0, stddev=0.01))
is_b3 = tf.Variable(tf.zeros([32]), name='is_b3')
is_w4 = tf.Variable(tf.random.normal([32, dims], name='is_w4', mean=0, stddev=0.01))
is_b4 = tf.Variable(tf.zeros([dims]), name='is_b4')



# trainable vars
i_g_para = [i_g_user_emb_matrix, i_g_item_emb_matrix, i_g_bias]
u_g_para = [u_g_user_emb_matrix, u_g_bias]
si_para = [si_w1, si_b1, si_w2, si_b2, si_w3, si_b3, si_w4, si_b4]
is_para = [is_w1, is_b1, is_w2, is_b2, is_w3, is_b3, is_w4, is_b4]
i_d_para = [i_d_user_emb_matrix, i_d_item_emb_matrix, i_d_bias]
u_d_para = [u_d_user_emb_matrix, u_d_bias]


si_trainable_vars = [u_g_user_emb_matrix] + si_para
is_trainable_vars = [i_g_user_emb_matrix] + is_para

i_g_trainable_variables = [i_g_item_emb_matrix, i_g_user_emb_matrix, i_g_bias] + si_para
u_g_trainable_variables = [u_g_user_emb_matrix, u_g_bias] + is_para

transfer_trainable_variables = si_trainable_vars + is_trainable_vars
i_d_trainable_variables = [i_d_user_emb_matrix, i_d_item_emb_matrix, i_d_bias]
u_d_trainable_variables = [u_d_user_emb_matrix, u_d_bias]

# chaeck-point
all_trainables = i_g_para + u_g_para + si_para + is_para + i_d_para + u_d_para
ckpt = tf.train.Checkpoint()
ckpt.listed = all_trainables

manager = tf.train.CheckpointManager(ckpt, 'tf_ckpts', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

def compute_l2_regularization(trainables, reg_para):
    res = 0
    for trainable_var in trainables:
        res += tf.nn.l2_loss(trainable_var)
    # print(res)
    return res * reg_para

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def simple_test_one_user(x):
    rating = x[0]
    u = x[1]

    test_items = list(dl.all_items - set(dl.user_pos_train[u]))
    item_score = []
    for i in test_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in dl.user_pos_val[u]:
            r.append(1)
        else:
            r.append(0)

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])
    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)

    return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])


# Item Discriminator
def item_discriminator_para_lookup(user_ids, pos_item_ids, neg_item_ids):
    i_d_user_embedding = tf.nn.embedding_lookup(i_d_user_emb_matrix, user_ids)
    i_d_pos_item_embedding = tf.nn.embedding_lookup(i_d_item_emb_matrix, pos_item_ids)
    i_d_neg_item_embedding = tf.nn.embedding_lookup(i_d_item_emb_matrix, neg_item_ids)
    i_d_pos_sbias = tf.gather(i_d_bias, pos_item_ids)
    i_d_neg_sbias = tf.gather(i_d_bias, neg_item_ids)
    return i_d_user_embedding, i_d_pos_item_embedding, i_d_neg_item_embedding, i_d_pos_sbias, i_d_neg_sbias

def item_discriminator_reward_for_gen(user_indices, item_indices):
    i_d_user_embedding = tf.nn.embedding_lookup(i_d_user_emb_matrix, user_indices)
    i_d_item_embedding = tf.nn.embedding_lookup(i_d_item_emb_matrix, item_indices)
    i_d_sbias = tf.gather(i_d_bias, item_indices)
    i_d_inner_prod = tf.reduce_sum(tf.multiply(i_d_user_embedding, i_d_item_embedding), axis=-1) + i_d_sbias
    return tf.math.log(tf.clip_by_value(1 + tf.math.exp(i_d_inner_prod), 1e-5, 1))


def item_discriminator_reward(i_d_user_embedding, i_d_item_embedding, i_d_sbias):
    i_d_inner_prod = tf.reduce_sum(tf.multiply(i_d_user_embedding, i_d_item_embedding), axis=-1) + i_d_sbias
    i_sigmoid = tf.math.sigmoid(i_d_inner_prod)
    return i_sigmoid


def item_discriminator_loss(user_indices, positive_item_indices, negative_item_indices):
    i_d_user_embedding, i_d_pos_item_embedding, i_d_neg_item_embedding, i_d_pos_sbias, i_d_neg_sbias = \
        item_discriminator_para_lookup(user_indices, positive_item_indices, negative_item_indices)

    positive_reward = item_discriminator_reward(i_d_user_embedding, i_d_pos_item_embedding, i_d_pos_sbias)
    negative_reward = item_discriminator_reward(i_d_user_embedding, i_d_neg_item_embedding, i_d_neg_sbias)

    i_d_loss = -tf.math.log(tf.clip_by_value(positive_reward, 1e-5, 1)) - tf.math.log(tf.clip_by_value(1 - negative_reward, 1e-5, 1)) + \
               compute_l2_regularization([i_d_user_embedding, i_d_user_embedding, i_d_pos_item_embedding, i_d_neg_item_embedding, i_d_pos_sbias, i_d_neg_sbias], config.dis_l2_reg)
    return i_d_loss / batch_size


def train_item_discriminator(user_indices, positive_item_indices, negative_item_indices):
    with tf.GradientTape() as tape:
        item_d_loss = item_discriminator_loss(user_indices, positive_item_indices, negative_item_indices)
        item_d_grad = tape.gradient(item_d_loss, i_d_trainable_variables)
        optimizer.apply_gradients(zip(item_d_grad, i_d_trainable_variables))
        return tf.reduce_sum(item_d_loss).numpy()


# User Discriminator
def user_discriminator_para_lookup(user_ids, pos_friend_ids, neg_friend_ids):
    u_d_user_embedding = tf.nn.embedding_lookup(u_d_user_emb_matrix, user_ids)
    u_d_pos_friend_embedding = tf.nn.embedding_lookup(u_d_user_emb_matrix, pos_friend_ids)
    u_d_neg_friend_embedding = tf.nn.embedding_lookup(u_d_user_emb_matrix, neg_friend_ids)
    u_d_pos_sbias = tf.gather(u_d_bias, pos_friend_ids)
    u_d_neg_sbias = tf.gather(u_d_bias, neg_friend_ids)
    return u_d_user_embedding, u_d_pos_friend_embedding, u_d_neg_friend_embedding, u_d_pos_sbias, u_d_neg_sbias


def user_discriminator_reward_for_gen(user_indices, friend_indices):
    u_d_user_embedding = tf.nn.embedding_lookup(u_d_user_emb_matrix, user_indices)
    u_d_friend_embedding = tf.nn.embedding_lookup(u_d_user_emb_matrix, friend_indices)
    u_d_sbias = tf.gather(u_d_bias, friend_indices)
    u_d_inner_prod = tf.reduce_sum(tf.multiply(u_d_user_embedding, u_d_friend_embedding), axis=-1) + u_d_sbias
    return tf.math.log(tf.clip_by_value(1 + tf.math.exp(u_d_inner_prod), 1e-5, 1))

def user_discriminator_reward(u_d_user_embedding, u_d_friend_embedding, u_d_sbias):
    u_d_inner_prod = tf.reduce_sum(tf.multiply(u_d_user_embedding, u_d_friend_embedding), axis=-1) + u_d_sbias
    u_sigmoid = tf.math.sigmoid(u_d_inner_prod)
    return u_sigmoid


def user_discriminator_loss(user_indices, positive_friend_indices, negative_friend_indices):
    u_d_user_embedding, u_d_pos_friend_embedding, u_d_neg_friend_embedding, u_d_pos_sbias, u_d_neg_sbias \
        = user_discriminator_para_lookup(user_indices, positive_friend_indices, negative_friend_indices)

    positive_reward = user_discriminator_reward(u_d_user_embedding, u_d_pos_friend_embedding, u_d_pos_sbias)
    negative_reward = user_discriminator_reward(u_d_user_embedding, u_d_neg_friend_embedding, u_d_neg_sbias)
    u_d_loss = -tf.math.log(tf.clip_by_value(positive_reward, 1e-5, 1)) - tf.math.log(tf.clip_by_value(1 - negative_reward, 1e-5, 1)) + \
               compute_l2_regularization([u_d_user_embedding, u_d_pos_friend_embedding, u_d_neg_friend_embedding, u_d_pos_sbias, u_d_neg_sbias], config.dis_l2_reg)
    return u_d_loss / batch_size


def train_user_discriminator(user_indices, positive_friend_indices, negative_friend_indices):
    with tf.GradientTape() as tape:
        user_d_loss = user_discriminator_loss(user_indices, positive_friend_indices, negative_friend_indices)
        user_d_grad = tape.gradient(user_d_loss, u_d_trainable_variables)
        optimizer.apply_gradients(zip(user_d_grad, u_d_trainable_variables))
        return tf.reduce_sum(user_d_loss).numpy()


# Social -> Item mapping
def get_emb_from_social_space(user_indices):
    u_g_user_embedding = tf.nn.embedding_lookup(u_g_user_emb_matrix, user_indices)
    return transfer_to_item_domain(u_g_user_embedding)

def transfer_to_item_domain(u_user_embeddings):
    si_layer1 = tf.nn.relu(tf.matmul(u_user_embeddings, si_w1) + si_b1)
    si_layer2 = tf.nn.relu(tf.matmul(si_layer1, si_w2) + si_b2)
    si_layer3 = tf.nn.relu(tf.matmul(si_layer2, si_w3) + si_b3)
    i_user_embedding_sim = tf.matmul(si_layer3, si_w4) + si_b4
    return i_user_embedding_sim


# Item -> Social mapping
def get_emb_from_item_space(user_indices):
    i_g_user_embedding = tf.nn.embedding_lookup(i_g_user_emb_matrix, user_indices)
    return transfer_to_social_domain(i_g_user_embedding)

def transfer_to_social_domain(i_user_embeddings):
    is_layer1 = tf.nn.relu(tf.matmul(i_user_embeddings, is_w1) + is_b1)
    is_layer2 = tf.nn.relu(tf.matmul(is_layer1, is_w2) + is_b2)
    is_layer3 = tf.nn.relu(tf.matmul(is_layer2, is_w3) + is_b3)
    u_user_embedding_sim = tf.matmul(is_layer3, is_w4) + is_b4
    return u_user_embedding_sim



# Item Generator
def item_generator_sample_prob(user_indices):
    i_g_user_embedding = get_emb_from_social_space(user_indices)
    user_all_item_logits = tf.matmul(i_g_user_embedding, i_g_item_emb_matrix, transpose_b=True) + i_g_bias
    user_all_item_softmax = tf.clip_by_value(tf.nn.softmax(user_all_item_logits/ 0.2), 1e-5, 1)
    dist = tfd.Categorical(probs=user_all_item_softmax)  # for each user, sample an item
    sampled_ids = dist.sample(sample_shape=[])
    sampled_log_prob = tf.math.log(tf.gather_nd(user_all_item_softmax, indices=list(enumerate(sampled_ids))))
    return sampled_ids, sampled_log_prob

def item_generator_sample_prob_importance_sampling(user_id, pos):
    sample_lambda = 0.2
    i_g_user_embedding = get_emb_from_social_space(user_id)
    user_all_item_logits = tf.reduce_sum(tf.multiply(i_g_user_embedding, i_g_item_emb_matrix), 1) + i_g_bias
    user_all_item_softmax_ori = tf.clip_by_value(tf.reshape(tf.nn.softmax(tf.reshape(user_all_item_logits, [1, -1])), [-1]), 1e-5, 1)

    # for sampling, we use another process which does the same thing (tf softmax has bugs)
    # user_all_item_exp_logits = np.exp(tf.reshape(user_all_item_logits, [-1]).numpy())
    # user_all_item_softmax = user_all_item_exp_logits / np.sum(user_all_item_exp_logits) + 1e-5   # we clip here
    user_all_item_softmax = user_all_item_softmax_ori.numpy()
    # importance sampling
    pn = (1 - sample_lambda) * user_all_item_softmax
    pn[pos] += sample_lambda / len(pos)
    pn = pn / np.sum(pn)

    sampled_items = np.random.choice(np.arange(n_items), 2 * len(pos), p=pn)

    sampled_prob_ori = tf.gather(user_all_item_softmax_ori, indices=sampled_items)
    sampled_prob = user_all_item_softmax[sampled_items]
    reward_reweights = sampled_prob / pn[sampled_items]
    sampled_log_prob = tf.math.log(sampled_prob_ori)
    return sampled_items, sampled_log_prob, reward_reweights

def train_item_generator(user_indices, all_positive_items):
    for user_id, positive_items in zip(user_indices, all_positive_items):
        # item loss computation
        with tf.GradientTape() as tape:
            user_id = tf.expand_dims(user_id, 0)
            sampled_ids, log_sampled_prob, reward_weight = item_generator_sample_prob_importance_sampling(user_id, positive_items)
            # get reward from discriminator
            user_ids = tf.tile(user_id, [len(positive_items)*2])
            i_reward_from_discriminator = item_discriminator_reward_for_gen(user_ids, sampled_ids)
            i_reward_from_discriminator = tf.multiply(i_reward_from_discriminator, reward_weight)


            regular_loss = regularization * tf.nn.l2_loss(get_emb_from_social_space(user_id) - tf.nn.embedding_lookup(i_g_user_emb_matrix, user_id))
            para_loss = config.gen_l2_reg * (tf.nn.l2_loss(tf.nn.embedding_lookup(i_g_item_emb_matrix, sampled_ids)) +
                                            tf.nn.l2_loss(tf.nn.embedding_lookup(i_g_user_emb_matrix, user_id)))

            i_g_loss = -tf.reduce_mean(tf.multiply(log_sampled_prob, i_reward_from_discriminator)) + regular_loss + para_loss  # + compute_l2_regularization(i_g_trainable_variables, 1e-3)

        grad = tape.gradient(i_g_loss, i_g_trainable_variables)
        optimizer.apply_gradients(zip(grad, i_g_trainable_variables))
    return tf.reduce_sum(i_g_loss).numpy() / batch_size

# User Generator
def user_generator_sample_prob(user_indices):
    u_g_user_embedding = get_emb_from_item_space(user_indices)
    user_all_friend_logits = tf.matmul(u_g_user_embedding, u_g_user_emb_matrix, transpose_b=True) + u_g_bias
    user_all_friend_softmax = tf.clip_by_value(tf.nn.softmax(user_all_friend_logits / 0.2), 1e-5, 1)
    dist = tfd.Categorical(probs=user_all_friend_softmax)  # for each user, sample an item
    sampled_ids = dist.sample(sample_shape=[])
    sampled_log_prob = tf.math.log(tf.gather_nd(user_all_friend_softmax, indices=list(enumerate(sampled_ids))))
    return sampled_ids, sampled_log_prob

def user_generator_sample_prob_importance_sampling(user_id, pos):
    sample_lambda = 0.2
    u_g_user_embedding = get_emb_from_item_space(user_id)
    user_all_friend_logits = tf.reduce_sum(tf.multiply(u_g_user_embedding, u_g_user_emb_matrix), 1) + u_g_bias
    user_all_friend_softmax_ori = tf.clip_by_value(tf.reshape(tf.nn.softmax(tf.reshape(user_all_friend_logits, [1, -1])), [-1]), 1e-5, 1)

    # for sampling, we use another process which does the same thing (tf softmax has bugs)
    user_all_friend_exp_logits = np.exp(tf.reshape(user_all_friend_logits, [-1]).numpy())
    user_all_friend_softmax = user_all_friend_exp_logits / np.sum(user_all_friend_exp_logits) + 1e-5   # we clip here
    # importance sampling
    pn = (1 - sample_lambda) * user_all_friend_softmax
    pn[pos] += sample_lambda / len(pos)
    pn = pn / np.sum(pn)
    sampled_friends = np.random.choice(np.arange(n_users), 2 * len(pos), p=pn)

    sampled_prob_ori = tf.gather(user_all_friend_softmax_ori, indices=sampled_friends)
    sampled_prob = user_all_friend_softmax[sampled_friends]
    reward_reweights = sampled_prob / pn[sampled_friends]
    sampled_log_prob = tf.math.log(sampled_prob_ori)
    return sampled_friends, sampled_log_prob, reward_reweights

def train_user_generator(user_indices, all_positive_friends):
    # user loss computation
    for user_id, positive_friends in zip(user_indices, all_positive_friends):
        # item loss computation
        with tf.GradientTape() as tape:
            user_id = tf.expand_dims(user_id, 0)
            sampled_ids, log_sampled_prob, reward_weight = user_generator_sample_prob_importance_sampling(user_id, positive_friends)
            # get reward from discriminator
            user_ids = tf.tile(user_id, [len(positive_friends)*2])
            u_reward_from_discriminator = user_discriminator_reward_for_gen(user_ids, sampled_ids)
            u_reward_from_discriminator = tf.multiply(u_reward_from_discriminator, reward_weight)

            regular_loss = regularization * tf.nn.l2_loss(get_emb_from_item_space(user_id) - tf.nn.embedding_lookup(u_g_user_emb_matrix, user_id))
            para_loss = config.gen_l2_reg * (tf.nn.l2_loss(tf.nn.embedding_lookup(u_g_user_emb_matrix, sampled_ids)) +
                                             tf.nn.embedding_lookup(u_g_user_emb_matrix, user_id))

            u_g_loss = -tf.reduce_sum(tf.multiply(log_sampled_prob, u_reward_from_discriminator)) + regular_loss + para_loss  # + compute_l2_regularization(u_g_trainable_variables, 1e-3)

        grad = tape.gradient(u_g_loss, u_g_trainable_variables)
        optimizer.apply_gradients(zip(grad, u_g_trainable_variables))
    return tf.reduce_sum(u_g_loss).numpy() / batch_size


def simple_test(model='gen'):
    result = np.array([0.] * 6)
    pool = multiprocessing.Pool(cores)
    batch_size = 128
    test_users = list(dict(dl.val_item).keys())
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + batch_size]
        index += batch_size

        if model == 'gen':
            user_embedding = get_emb_from_social_space(user_batch)
            user_batch_rating = tf.matmul(user_embedding, i_g_item_emb_matrix, transpose_b=True) + i_g_bias
        else:
            user_embedding = tf.nn.embedding_lookup(i_d_user_emb_matrix, user_batch)
            user_batch_rating = tf.matmul(user_embedding, i_d_item_emb_matrix, transpose_b=True) + i_d_bias

        user_batch_rating_uid = zip(user_batch_rating.numpy(), user_batch)
        batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret



if __name__=="__main__":
    report_fr = 10

    for circle in range(15):
        # Train item gan
        dl.set_data('item')

        # train item discriminator
        dl.epoch_cnt = 0
        previous_epoch = 0
        batch_cnt = 0
        batch_losses = []
        previous_batch_loss = 100
        while dl.epoch_cnt < 15:
            batch = next(dl)
            user_ids, positive_items = tf.stack(batch, axis=1)
            negative_items, _ = item_generator_sample_prob(user_ids)
            loss = train_item_discriminator(user_ids, positive_items, negative_items)
            batch_losses.append(loss)
            if previous_epoch < dl.epoch_cnt:
                save_path = manager.save()
                print("Saved checkpoint for item_discriminator step {}: {}".format(dl.epoch_cnt, save_path))

                mean_batch_loss = float(np.mean(batch_losses))
                batch_losses = list()
                previous_epoch = dl.epoch_cnt
                print("average loss at epoch %d is : %f" % (dl.epoch_cnt, mean_batch_loss))
                print('l2_loss: %f' % compute_l2_regularization(i_d_trainable_variables, 1))
                # if np.abs((previous_batch_loss - mean_batch_loss) / mean_batch_loss) < 0.001:
                #     break
                previous_batch_loss = mean_batch_loss
            if batch_cnt % report_fr == 0:
                print(loss)
                print('l2_loss: %f' % compute_l2_regularization(i_d_trainable_variables, 1))
            batch_cnt += 1
        print("end of item disc")
        log = simple_test('dis')
        print("dis " + str(log))
        with open('log.txt', 'a') as f:
            f.write(str(circle) + '\tdis\t' + str(log) + '\n')

        # train item generator
        dl.epoch_cnt = 0
        previous_epoch = 0
        previous_batch_loss = 100
        batch_cnt = 0
        batch_losses = []
        while dl.epoch_cnt < 15:
            batch = next(dl)
            all_positive_items = [dl.user_pos_train[u] for u, _ in batch]
            user_ids, _ = tf.stack(batch, axis=1)
            loss = train_item_generator(user_ids, all_positive_items)
            batch_losses.append(loss)
            if previous_epoch < dl.epoch_cnt:
                save_path = manager.save()
                print("Saved checkpoint for item_generator step {}: {}".format(dl.epoch_cnt, save_path))
                mean_batch_loss = float(np.mean(batch_losses))
                batch_losses = list()
                previous_epoch = dl.epoch_cnt
                print("loss at epoch %d is : %f" % (dl.epoch_cnt, mean_batch_loss))
                # if np.abs((previous_batch_loss - mean_batch_loss) / mean_batch_loss) < 0.01:
                #     break
                previous_batch_loss = mean_batch_loss
            print(loss)
            print('l2_loss: %f' % compute_l2_regularization(i_g_trainable_variables, 1))
            batch_cnt += 1
            # if loss < 1e-3:
            #     break
        print("end of item gene")
        log = simple_test('gen')
        print("gen " + str(log))
        with open('log.txt', 'a') as f:
            f.write(str(circle) + '\tgen\t' + str(log) + '\n')

        # train user gan
        dl.set_data('user')

        # train user discriminator
        dl.epoch_cnt = 0
        previous_epoch = 0
        batch_cnt = 0
        batch_losses = []
        previous_batch_loss = 100
        while dl.epoch_cnt < 10:
            batch = next(dl)
            user_ids, positive_friends = tf.stack(batch, axis=1)
            negative_friends, _ = user_generator_sample_prob(user_ids)
            loss = train_user_discriminator(user_ids, positive_friends, negative_friends)
            batch_losses.append(loss)
            if previous_epoch < dl.epoch_cnt:
                save_path = manager.save()
                print("Saved checkpoint for user_discriminator step {}: {}".format(dl.epoch_cnt, save_path))

                mean_batch_loss = float(np.mean(batch_losses))
                batch_losses = list()
                previous_epoch = dl.epoch_cnt
                print("average loss at epoch %d is : %f" % (dl.epoch_cnt, mean_batch_loss))
                # if np.abs((previous_batch_loss - mean_batch_loss) / mean_batch_loss) < 0.01:
                #     break
                previous_batch_loss = mean_batch_loss
            if batch_cnt % report_fr == 0:
                print(loss)
                print('l2_loss: %f' % compute_l2_regularization(u_d_trainable_variables, 1))
            batch_cnt += 1
        print("end of user disc")

        # trian user generator
        dl.epoch_cnt = 0
        previous_epoch = 0
        loss = 10
        batch_cnt = 0
        batch_losses = []
        previous_batch_loss = 100
        while dl.epoch_cnt < 10:
            batch = next(dl)
            all_positive_friends = [dl.user_friends[u] for u, _ in batch]
            user_ids, _ = tf.stack(batch, axis=1)
            loss = train_user_generator(user_ids, all_positive_friends)
            batch_losses.append(loss)
            if previous_epoch < dl.epoch_cnt:
                save_path = manager.save()
                print("Saved checkpoint for user_generator step {}: {}".format(dl.epoch_cnt, save_path))
                mean_batch_loss = float(np.mean(batch_losses))
                batch_losses = list()
                previous_epoch = dl.epoch_cnt
                print("loss at epoch %d is : %f" % (dl.epoch_cnt, mean_batch_loss))
                # if np.abs((previous_batch_loss - mean_batch_loss) / mean_batch_loss) < 0.01:
                #     break
                previous_batch_loss = mean_batch_loss
            # if loss < 1e-3:
            #     break
            print('l2_loss: %f' % compute_l2_regularization(u_g_trainable_variables, 1))
            print(loss)
            batch_cnt += 1
        print("end of user gen")





