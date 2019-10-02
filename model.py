import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import RMSprop
from data_fetcher import data_loader

tf.random.set_seed(3615)
tfd = tfp.distributions

dims = 200
batch_size = 128
regularization = 100
dl = data_loader(batch_size, 'item')

n_users = dl.n_user
n_items = dl.n_item

# user_indices = [1, 3, 5]
# positive_item_indices = [0, 3, 8]
# negative_item_indices = [2, 9, 10]

optimizer = RMSprop(0.01)
initdelta = 0.05


# positive_friend_indices = [2, 4, 3]
# negative_friend_indices = [0, 0, 0]


# item space generator
i_g_user_emb_matrix = tf.Variable(tf.random.uniform([n_users, dims], name='i_g_user', minval=-initdelta, maxval=initdelta))
i_g_item_emb_matrix = tf.Variable(tf.random.uniform([n_items, dims], name='i_g_item', minval=-initdelta, maxval=initdelta))
i_g_bias = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='i_g_bias')

# user space generator
u_g_user_emb_matrix = tf.Variable(tf.random.uniform([n_users, dims], name='u_g_user', minval=-initdelta, maxval=initdelta))
u_g_bias = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='u_g_bias')

# social space to item space mapping
si_w1 = tf.Variable(tf.random.normal([dims, 1024], name='si_w1', mean=0, stddev=0.01))
si_b1 = tf.Variable(tf.zeros([1024]), name='si_b1')
si_w2 = tf.Variable(tf.random.normal([1024, 512], name='si_w2', mean=0, stddev=0.01))
si_b2 = tf.Variable(tf.zeros([512]), name='si_b2')
si_w3 = tf.Variable(tf.random.normal([512, dims], name='si_w3', mean=0, stddev=0.01))
si_b3 = tf.Variable(tf.zeros([dims]), name='si_b3')

# item space to social space mapping
is_w1 = tf.Variable(tf.random.normal([dims, 1024], name='is_w1', mean=0, stddev=0.01))
is_b1 = tf.Variable(tf.zeros([1024]), name='is_b1')
is_w2 = tf.Variable(tf.random.normal([1024, 512], name='is_w2', mean=0, stddev=0.01))
is_b2 = tf.Variable(tf.zeros([512]), name='is_b2')
is_w3 = tf.Variable(tf.random.normal([512, dims], name='is_w3', mean=0, stddev=0.01))
is_b3 = tf.Variable(tf.zeros([dims]), name='is_b3')

# item space discriminator
i_d_user_emb_matrix = tf.Variable(tf.random.uniform([n_users, dims], name='i_d_user', minval=-initdelta, maxval=initdelta))
i_d_item_emb_matrix = tf.Variable(tf.random.uniform([n_items, dims], name='i_d_item', minval=-initdelta, maxval=initdelta))
i_d_bias = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='i_d_bias')


# user space discriminator
u_d_user_emb_matrix = tf.Variable(tf.random.uniform([n_users, dims], name='u_d_user', minval=-initdelta, maxval=initdelta))
u_d_bias = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='u_d_bias')


# trainable vars
si_trainable_vars = [u_g_user_emb_matrix, si_w1, si_b1, si_w2, si_b2, si_w3, si_b3]
is_trainable_vars = [i_g_item_emb_matrix, is_w1, is_b1, is_w2, is_b2, is_w3, is_b3]
i_g_trainable_variables = [i_g_item_emb_matrix, i_g_bias] + si_trainable_vars
u_g_trainable_variables = [u_g_user_emb_matrix, u_g_bias] + is_trainable_vars
transfer_trainable_variables = si_trainable_vars + is_trainable_vars
i_d_trainable_variables = [i_d_user_emb_matrix, i_d_item_emb_matrix, i_d_bias]
u_d_trainable_variables = [u_d_user_emb_matrix, u_d_bias]




# item discriminator
def item_discriminator_reward(user_indices, item_indices):
    i_d_user_embedding = tf.nn.embedding_lookup(i_d_user_emb_matrix, user_indices)
    i_d_item_embedding = tf.nn.embedding_lookup(i_d_item_emb_matrix, item_indices)
    i_d_inner_prod = tf.reduce_sum(tf.multiply(i_d_user_embedding, i_d_item_embedding), axis=-1) + i_d_bias
    i_sigmoid = tf.math.sigmoid(i_d_inner_prod)
    return i_sigmoid

def item_discriminator_loss(user_indices, positive_item_indices, negative_item_indices):
    positive_reward = item_discriminator_reward(user_indices, positive_item_indices)
    negative_reward = item_discriminator_reward(user_indices, negative_item_indices)
    i_d_loss = -tf.math.log(positive_reward+1e-7) - tf.math.log(1 - negative_reward + 1e-7)
    return i_d_loss / batch_size

def train_item_discriminator(user_indices, positive_item_indices, negative_item_indices):
    with tf.GradientTape() as tape:
        item_d_loss = item_discriminator_loss(user_indices, positive_item_indices, negative_item_indices)
        item_d_grad = tape.gradient(item_d_loss, i_d_trainable_variables)
        optimizer.apply_gradients(zip(item_d_grad, i_d_trainable_variables))
        return tf.reduce_sum(item_d_loss).numpy()

# user discriminator
def user_discriminator_reward(user_indices, friend_indices):
    u_d_user_embedding = tf.nn.embedding_lookup(u_d_user_emb_matrix, user_indices)
    u_d_friend_embedding = tf.nn.embedding_lookup(u_d_user_emb_matrix, friend_indices)
    u_d_inner_prod = tf.reduce_sum(tf.multiply(u_d_user_embedding, u_d_friend_embedding), axis=-1) + u_d_bias
    u_sigmoid = tf.math.sigmoid(u_d_inner_prod)
    return u_sigmoid

def user_discriminator_loss(user_indices, positive_friend_indices, negative_friend_indices):
    positive_reward = user_discriminator_reward(user_indices, positive_friend_indices)
    negative_reward = user_discriminator_reward(user_indices, negative_friend_indices)
    u_d_loss = -tf.math.log(positive_reward + 1e-7) - tf.math.log(1 - negative_reward + 1e-7)
    return u_d_loss

def train_user_discriminator(user_indices, positive_friend_indices, negative_friend_indices):
    with tf.GradientTape() as tape:
        user_d_loss = user_discriminator_loss(user_indices, positive_friend_indices, negative_friend_indices)
        user_d_grad = tape.gradient(user_d_loss, u_d_trainable_variables)
        optimizer.apply_gradients(zip(user_d_grad, u_d_trainable_variables))
        return tf.reduce_sum(user_d_loss).numpy()


# social -> item mapping
def get_emb_from_social_space(user_indices):
    u_g_user_embedding = tf.nn.embedding_lookup(u_g_user_emb_matrix, user_indices)
    return transfer_to_item_domain(u_g_user_embedding)

def transfer_to_item_domain(u_user_embeddings):
    si_layer1 = tf.nn.relu(tf.matmul(u_user_embeddings, si_w1) + si_b1)
    si_layer2 = tf.nn.relu(tf.matmul(si_layer1, si_w2) + si_b2)
    i_user_embedding_sim = tf.nn.relu(tf.matmul(si_layer2, si_w3) + si_b3)
    return i_user_embedding_sim



# item -> social mapping
def get_emb_from_item_space(user_indices):
    i_g_user_embedding = tf.nn.embedding_lookup(i_g_user_emb_matrix, user_indices)
    return transfer_to_social_domain(i_g_user_embedding)

def transfer_to_social_domain(i_user_embeddings):
    is_layer1 = tf.nn.relu(tf.matmul(i_user_embeddings, is_w1) + is_b1)
    is_layer2 = tf.nn.relu(tf.matmul(is_layer1, is_w2) + is_b2)
    u_user_embedding_sim = tf.nn.relu(tf.matmul(is_layer2, is_w3) + is_b3)
    return u_user_embedding_sim


def train_transfer_component(user_indices):
    # transfer loss
    with tf.GradientTape() as tape:
        u_g_user_embedding = tf.nn.embedding_lookup(u_g_user_emb_matrix, user_indices)
        i_g_user_embedding = tf.nn.embedding_lookup(i_g_user_emb_matrix, user_indices)

        regular_loss = regularization * (tf.nn.l2_loss(
            transfer_to_social_domain(transfer_to_item_domain(u_g_user_embedding)) - u_g_user_embedding) +
                                         tf.nn.l2_loss(transfer_to_item_domain(
                                             transfer_to_social_domain(i_g_user_embedding)) - i_g_user_embedding))
        # apply gradients
        grad = tape.gradient(regular_loss, transfer_trainable_variables)
        optimizer.apply_gradients(zip(grad, transfer_trainable_variables))


# item generator
def item_generator_sample_prob(user_indices):
    i_g_user_embedding = get_emb_from_social_space(user_indices)
    user_all_item_logits = tf.matmul(i_g_user_embedding, i_g_item_emb_matrix, transpose_b=True) + i_g_bias
    user_all_item_softmax = tf.nn.softmax(user_all_item_logits)
    dist = tfd.Categorical(probs=user_all_item_softmax)  # for each user, sample an item
    sampled_ids = dist.sample(sample_shape=[])
    sampled_log_prob = tf.gather_nd(user_all_item_softmax, indices=list(enumerate(sampled_ids)))
    return sampled_ids, sampled_log_prob


def train_item_generator(user_indices):
    # item loss computation
    with tf.GradientTape(persistent=True) as tape:
        sampled_ids, log_sampled_prob = item_generator_sample_prob(user_indices)
        # get reward from discriminator
        i_reward_from_discriminator = item_discriminator_reward(user_indices, sampled_ids)
        # compute weights for each loss
        i_g_loss_weights = tf.math.log(1 - i_reward_from_discriminator)
        # print(i_reward_from_discriminator)
        # print(log_sampled_prob)

        for loss, loss_weight in zip(log_sampled_prob, i_g_loss_weights):
            grad = tape.gradient(loss, i_g_trainable_variables)
            grad = [tf.multiply(g, loss_weight) for g in grad]
            optimizer.apply_gradients(zip(grad, i_g_trainable_variables))


# user generator
def user_generator_sample_prob(user_indices):
    u_g_user_embedding = get_emb_from_item_space(user_indices)
    user_all_friend_logits = tf.matmul(u_g_user_embedding, u_g_user_emb_matrix, transpose_b=True) + u_g_bias
    user_all_friend_softmax = tf.nn.softmax(user_all_friend_logits)
    dist = tfd.Categorical(probs=user_all_friend_softmax)  # for each user, sample an item
    sampled_ids = dist.sample(sample_shape=[])
    sampled_log_prob = tf.gather_nd(user_all_friend_softmax, indices=list(enumerate(sampled_ids)))
    return sampled_ids, sampled_log_prob


def train_user_generator(user_indices):
    # user loss computation
    with tf.GradientTape(persistent=True) as tape:
        sampled_ids, log_sampled_prob = user_generator_sample_prob(user_indices)
        # get reward from discriminator
        u_reward_from_discriminator = user_discriminator_reward(user_indices, sampled_ids)
        # compute weights for each loss
        u_g_loss_weights = tf.math.log(1 - u_reward_from_discriminator)
        print(u_reward_from_discriminator)
        print(log_sampled_prob)

        for loss, loss_weight in zip(log_sampled_prob, u_g_loss_weights):
            grad = tape.gradient(loss, u_g_trainable_variables)
            grad = [tf.multiply(g, loss_weight) for g in grad]
            optimizer.apply_gradients(zip(grad, u_g_trainable_variables))



if __name__=="__main__":
    # train item discriminator
    dl.set_data('item')
    while dl.epoch_cnt < 10:
        batch = next(dl)
        user_ids, positive_items = tf.stack(batch, axis=1)
        negative_items, _ = item_generator_sample_prob(user_ids)
        loss = train_item_discriminator(user_ids, positive_items, negative_items)
        print(dl.epoch_cnt)
        print(loss)
    # train item generator

    # train user discriminator

    # trian user generator

    # train cross-unit





