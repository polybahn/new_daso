import math
import random
random.seed(3615)

class data_loader:
    def __init__(self, batch_size, d_flag="item"):
        self.epoch_cnt = 0
        self.d_flag = d_flag
        self.batch_size = batch_size
        base_path = 'data/ciao/'
        rating_path = base_path + 'rating.csv'
        social_path = base_path + 'social.csv'

        self.ratings = self.read_f(rating_path)
        self.socials = self.read_f(social_path)

        self.ori_items = sorted(set([i for u, i in self.ratings]))
        self.item_mapping = list(enumerate(self.ori_items))
        self.r_i_m = dict([(v, k) for k, v in self.item_mapping])
        self.write_mapping(base_path + 'item_mapping_new_to_ori.csv', self.item_mapping)
        self.n_item = len(self.ori_items)
        self.all_items = set(self.r_i_m.values())

        self.ori_users = set([u for u, i in self.ratings])
        self.ori_users.update(set(dict(self.socials).keys()))
        self.ori_users.update(set(dict(self.socials).values()))
        self.ori_users = sorted(self.ori_users)
        self.user_mapping = list(enumerate(self.ori_users))
        self.r_u_m = dict([(v, k) for k, v in self.user_mapping])
        self.write_mapping(base_path + 'user_mapping_new_to_ori.csv', self.user_mapping)
        self.n_user = len(self.ori_users)

        self.ratings = [(self.r_u_m[u], self.r_i_m[i]) for u, i in self.ratings]
        self.socials = [(self.r_u_m[u], self.r_u_m[f]) for u, f in self.socials]

        self.write_mapping(base_path + 'new_rating.csv', self.ratings)
        self.write_mapping(base_path + 'new_social.csv', self.socials)




        random.shuffle(self.ratings)
        random.shuffle(self.socials)

        self.train_item, self.val_item, self.test_item = self.train_val_test_split(self.ratings)

        # build for importance sampling to train generator
        self.user_pos_train = {}
        for u, i in self.train_item:
            if u not in self.user_pos_train:
                self.user_pos_train[u] = list()
            self.user_pos_train[u].append(i)

        self.user_friends = {}
        for u, f in self.socials:
            if u not in self.user_friends:
                self.user_friends[u] = list()
            if f not in self.user_friends:
                self.user_friends[f] = list()
            self.user_friends[u].append(f)
            self.user_friends[f].append(u)

        # build test positive items for precision and ndcg
        self.user_pos_val = {}
        for u, i in self.val_item:
            if u not in self.user_pos_val:
                self.user_pos_val[u] = set()
            self.user_pos_val[u].add(i)

        # mark position of current data iterator
        self.i_pos = 0
        self.u_pos = 0

    def write_mapping(self, fp, d):
        with open(fp, 'w') as f:
            for k, v in d:
                f.write(str(k) + '\t' + str(v) + '\n')

    def read_f(self, fp):
        tuple_l = list()
        with open(fp, 'r') as f:
            for line in f:
                eles = [int(e.strip()) for e in line.split(',')]
                tuple_l.append((eles[0], eles[1]))
        return tuple_l

    def train_val_test_split(self, data):
        len_data = len(data)
        train = data[: math.floor(len_data * 0.8)]
        val = data[math.floor(len_data * 0.8): math.floor(len_data * 0.9)]
        test = data[math.floor(len_data * 0.9):]
        return train, val, test

    def set_data(self, flag):
        self.d_flag = flag

    def __next__(self):
        temp_l = list()
        if self.d_flag == 'item':
            if self.i_pos + self.batch_size > len(self.train_item):
                self.epoch_cnt += 1
                temp_l += self.train_item[self.i_pos:]
                temp_l += self.train_item[:self.i_pos + self.batch_size - len(self.train_item)]
                self.i_pos = self.i_pos + self.batch_size - len(self.train_item)
                return temp_l
            else:
                temp_l = self.train_item[self.i_pos: self.i_pos + self.batch_size]
                self.i_pos += self.batch_size
        elif self.d_flag == 'user':
            if self.u_pos + self.batch_size > len(self.socials):
                self.epoch_cnt += 1
                temp_l += self.socials[self.u_pos:]
                temp_l += self.socials[:self.u_pos + self.batch_size - len(self.socials)]
                self.u_pos = self.u_pos + self.batch_size - len(self.socials)
                return temp_l
            else:
                temp_l = self.socials[self.u_pos: self.u_pos + self.batch_size]
                self.u_pos += self.batch_size
        return temp_l


if __name__=="__main__":
    ldr = data_loader(50, 'user')
    # while True:
    #     print(next(ldr))
