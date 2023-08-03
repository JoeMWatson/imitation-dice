import os
import pickle

envs = [
    'Hopper-v2'
]

n_demos = [
    30
]

n_seeds = 1

RES = {
    e: np.zeros((n_seeds, len(n_demos), 100))
}

for env in envs:
    for nd in n_demos:
        for s in range(n_seeds):
            path = os.path.join(
                'checkpoint_imitator', 'demodice', env,
                 f"expert-v2_5_['expert-v2', 'full_replay-v2']_[{nd}, 1000]")
            filename = os.path.join(path, f'{s}.pickle')
            with open(filename, 'rb') as f:
                res = pickle.load(f)
            iter = res['training_info']['iteration']
            rets = [res['training_info']['logs'][i]['log']['eval'] for i in range(len(res['training_info']['logs']))]
