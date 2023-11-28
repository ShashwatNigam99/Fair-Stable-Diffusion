import numpy as np
import glob
from deepface import DeepFace
from scipy.stats import entropy

fname = "/content/drive/MyDrive/Colab Notebooks/out"


def gen_res(model, attr='gender', fact='Man'):
    file_names = glob.glob(fname + "/*")
    m_dict, fm_dict = {}, {}
    k_list = []
    for this_f in file_names:
        this_url = this_f + model
        img_list = glob.glob(this_url + '*.png')
        print('\n', this_url, img_list)
        m, fm = 0, 0
        for this_img in img_list:
            objs = DeepFace.analyze(img_path=this_img, actions=[attr], enforce_detection=False)[0]
            objs_attr = objs[attr]
            tmp = max(objs_attr, key=objs_attr.get)
            if tmp == fact:
                m += 1
            else:
                fm += 1

        k = this_f.rpartition(' ')[-1]
        print('\n', k, m, fm)
        m_dict[k] = m
        fm_dict[k] = fm

    return m_dict, fm_dict


def run(**kwargs):
    m_stablediff, fm_stablediff = gen_res('/stablediff/', attr=kwargs['attr'], fact=kwargs['fact'])
    m_fairdiff, fm_fairdiff = gen_res('/fairdiff/', attr=kwargs['attr'], fact=kwargs['fact'])

    m_percentage_stable = sum(m_stablediff.values()) / (sum(m_stablediff.values()) + sum(fm_stablediff.values()))
    fm_percentage_stable = 1. - m_percentage_stable

    m_percentage_fair = sum(m_fairdiff.values()) / (sum(m_fairdiff.values()) + sum(fm_fairdiff.values()))
    fm_percentage_fair = sum(fm_fairdiff.values()) / (sum(m_fairdiff.values()) + sum(fm_fairdiff.values()))

    stablediff_kl = entropy(np.array(list(fm_stablediff.values())).astype(np.float),
                            np.array(list(m_stablediff.values())).astype(np.float))
    fairdiff_kl = entropy(np.array(list(fm_fairdiff.values())).astype(np.float),
                          np.array(list(m_fairdiff.values())).astype(np.float))

    stable_dict = {'m_percentage': m_percentage_stable, 'fm_percentage': fm_percentage_stable, 'kl': stablediff_kl}
    fair_dict = {'m_percentage': m_percentage_fair, 'fm_percentage': fm_percentage_fair, 'kl': fairdiff_kl}

    return stable_dict, fair_dict


def main():
    target_dict = {'gender': {'attr': 'gender', 'fact': 'Man'},
                   'race': {'attr': 'race', 'fact': 'white'}}

    final_res = {}

    for k, v in target_dict.items():
        stable_dict, fair_dict = run(**v)
        final_res[k] = {'stable': stable_dict, 'fair': fair_dict}

    print(final_res)
