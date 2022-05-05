import re
import matplotlib.pyplot as plt

def parse_log(filepath = '../es/logs/accuracy.log') -> dict[dict[float]]:
    data = {}
    with open(filepath) as f:
        lc = 0
        line = f.readline()
        start = False        
        acc_dict = {} 
        acc_list = []
        while line:
            run = "INFO:es:run:"
            if run in line:
                start = True
                run_num = line[(len(run)+1):-1]
                data[run_num] = {}
            else:
                if lc % 4 == 0:
                    acc_list.append(acc_dict)
                    acc_dict = {}
                if "accuracy" in line.lower():
                    key = line[(len("INFO:es:")):21].strip()
                    value = float(re.findall("\d+\.\d+",line)[0])
                    acc_dict[key] = value
                if start:
                    lc += 1

            line = f.readline()

        for _ in range(len(acc_list)):
            try:
                acc_list.remove({})
            except:
                break
        for i, k in enumerate(data):
            data[k] = acc_list[i]
    return data
            


def reformat_data(data) -> tuple[list[float], dict[list[float]]]:
    runs = []
    acc_dict = {}
    for k in data[list(data.keys())[0]]:
        acc_dict[k] = []
    for run in data:
        runs.append(run)
        for k in acc_dict:
            try:
                acc_dict[k].append(data[run][k])
            except:
                print("k" + str(k))
                print("data[run]" + str(data[run]))
                print("acc_dict[k]" + str(acc_dict[k]))
            

    return (runs, acc_dict)



def plot_acc(data_tuple):
    print("Plotting data")
    runs, data = data_tuple
    cmap = plt.cm.get_cmap("tab10", 4)
    for i, acc_label in enumerate(data):
        y = data[acc_label]
        plt.plot(runs, y, color=cmap(i), label=acc_label)
    
    plt.xlabel("Run number")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over runs")

    plt.legend()
    plt.savefig("../images/accuracy_run_plot.png")
    plt.show()


def main():
    data = reformat_data(parse_log())
    plot_acc(data)

main()
