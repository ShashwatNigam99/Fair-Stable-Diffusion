
def get_all_elem(f_name: str):
    output = []
    with open(f_name, 'r') as f:
        for line in f:
            output.append(line[:-1])
    return output

def gen_prompt(a1, a2, o1):
    l1 = get_all_elem(f'target_social_attributes/{a1}.txt')
    l2 = get_all_elem(f'target_social_attributes/{a2}.txt')

    print(l1)
    print(l2)

    import itertools
    comb = list(itertools.product(l1, l2))

    with open(f'target_social_attributes/{o1}.txt', 'w') as f:
        for elem in comb:
            prompt = f"Photo portrait of a {elem[0]} {elem[1]}"
            f.write(prompt + '\n')


gen_prompt('Ethnicity', 'Gender', 'prompt_social')
gen_prompt('Adjective', 'Profession', 'prompt_target')
