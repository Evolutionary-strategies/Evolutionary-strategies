import json
import os

epsilons = [0.005, 0.01, 0.02]

def get_data(path) -> dict[dict[list[float]]]:
    data = {}
    for file in os.listdir(path):
        filename = os.path.join(path, file)
        f = open(filename)
        model_data = json.load(f)
        data[file] = model_data
    return data
    

def print_table_start(attack):
    f_attacks2attack = {
        "LinfFastGradientAttack(rel_stepsize=1.0, abs_stepsize=None, steps=1, random_start=False)": "\(L_\infty\) FGSM",
        "LinfProjectedGradientDescentAttack(rel_stepsize=0.03333333333333333, abs_stepsize=None, steps=40, random_start=True)": "\(L_\infty\) PGD",
        "LinfAdditiveUniformNoiseAttack()": "\(L_\infty\) AUN",
        "LinfDeepFoolAttack(steps=50, candidates=10, overshoot=0.02, loss=logits)": "\(L_\infty\) DeepFool",
        "LinfBasicIterativeAttack(rel_stepsize=0.2, abs_stepsize=None, steps=10, random_start=False)": "\(L_\infty\) BIM",
        "L2ContrastReductionAttack(target=0.5)": "\(L_2\) CR",
        "L2AdditiveGaussianNoiseAttack()": "\(L_2\) AGN",
        "L2FastGradientAttack(rel_stepsize=1.0, abs_stepsize=None, steps=1, random_start=False)": "\(L_2\) FGSM",
        "L2BasicIterativeAttack(rel_stepsize=0.2, abs_stepsize=None, steps=10, random_start=False)": "\(L_2\) BIM",
        "L2DeepFoolAttack(steps=50, candidates=10, overshoot=0.02, loss=logits)": "\(L_2\) DeepFool",
        "L2ProjectedGradientDescentAttack(rel_stepsize=0.025, abs_stepsize=None, steps=50, random_start=True)": "\(L_2\) PGD"
    }
    print("\\begin{center}")
    print("\\begin{tabular}{ |c|ccc| }")
    print("\\hline")
    columns = str(len(epsilons) + 1)
    print("\multicolumn{"+ columns +"}{|c|}{" + f_attacks2attack[attack] + "} \\\\")
    print("\\hline")
    line = "Model"
    for eps in epsilons:
        line += " & epsilon " + str(eps)
    line += "\\\\"
    print(line)
    print("\\hline")

def print_table_end():
    print("\end{tabular}")
    print("\end{center}")

def print_line(model, attack_acc):
    model_number = {
        "accuracy_data_nes_model_sigma01.json": "1",
        "accuracy_data_nes_model_sigma015.json": "2",
        "accuracy_data_es_model_sigma015_acc073_1.json": "3",
        "accuracy_data_es_model_sigma015_acc073_2.json": "4"
    }
    line = model_number[model]
    for acc in attack_acc:
        acc = round(acc*100, 2)
        line += " & " + str(acc) + " \%"
    line += " \\\\"
    line += "\n\\hline"
    return line


def main():
    path = "../aa_results"
    data = get_data(path)

    first_key = list(data.keys())[0]

    for attack in data[first_key]:
        print_table_start(attack)
        models = data.keys()
        lines = []
        for model in models:
            line = print_line(model, data[model][attack])
            lines.append(line)
        for line in lines:
            if type(int(line[0:1])) != type(1):
                print(line)
        for j in range(len(lines)):
            for i in range(len(lines)):
                # print(lines[i][0:1])
                if str(j+1) == lines[i][0:1]:
                    print(lines[i])
                    # lines.remove(lines[i])
        print_table_end()

main()