from fileinput import filename
import matplotlib.pyplot as plt

epsilons = [
    [0.0, 0.3, 0.5, 1.0, 3.0, 5.0, 8.0, 10.0, 13.0, 15.0, 18.0, 20.0],
    [0.0, 0.3, 0.5, 1.0, 3.0, 5.0, 8.0, 10.0, 13.0, 15.0, 18.0, 20.0],
    [0.0, 0.005, 0.01, 0.02],
    [0.0, 0.005, 0.01, 0.02],
    [0.0, 0.005, 0.01, 0.02],
    [0.0, 0.01, 0.03],
    [0.0, 0.01, 0.02, 0.03],
    [0.0, 0.03, 0.05, 0.1],
    [0.0, 0.0001, 0.3, 0.5, 1.0],
    [0.0, 0.0001, 0.001, 0.005, 0.1, 0.3, 0.5],
    [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 3.0, 5.0],
    [0.0, 0.005, 0.01, 0.02, 0.1, 0.3, 0.5, 0.8, 1.0]
]
data = {
    "AGN $L_2$": 
    {
        "Gradient Descent": [54.48, 54.61, 54.53, 54.59, 52.91, 49.93, 41.81, 34.28, 24.43, 20.44, 16.49, 15.52],
        "1": [54.79, 54.75, 54.88, 54.68, 52.02, 43.97, 31.36, 26.27, 20.71, 18.74, 15.9, 14.97],
        "2": [54.93, 54.99, 54.85, 55.1, 54.56, 50.7, 43.22, 37.84, 30.23, 26.18, 20.72, 18.58],
        "3": [54.50, 54.66, 54.6, 54.29, 53.4, 50.56, 43.36, 37.92, 31.32, 27.1, 22.7, 20.38],
        "4": [51.04, 50.95, 51.0, 50.87, 49.3, 46.12, 38.94, 34.21, 27.33, 23.54, 18.97, 16.6]
    },
    "CR $L_2$": 
    {
        "Gradient Descent": [54.48, 54.56, 54.53, 54.55, 52.44, 45.66, 29.44, 20.37, 13.12, 11.09, 10.25, 10.06],
        "1": [54.79, 54.63, 54.55, 54.47, 53.08, 48.46, 35.22, 25.06, 15.15, 12.31, 10.77, 10.31],
        "2": [54.93, 54.81, 54.83, 54.86, 51.64, 45.48, 30.25, 21.81, 14.57, 12.09, 10.47, 10.14],
        "3": [54.50, 54.49, 54.5, 54.51, 53.04, 47.88, 32.82, 23.03, 14.4, 11.88, 10.55, 10.14],
        "4": [51.04, 51.09, 51.17, 50.9, 48.83, 44.05, 30.8, 22.51, 15.32, 12.85, 10.91, 10.53]
    },
    "FGSM CE $L_\infty$":
    {
        "Gradient Descent": [54.48, 44.37, 35.19, 20.59],
        "1": [54.79, 54.75, 54.75, 54.75],
        "2": [54.93, 54.66, 54.66, 54.66],
        "3": [54.50, 54.15, 54.15, 54.15],
        "4": [51.04, 50.93, 50.93, 50.93]
    },
    "FGSM CW $L_\infty$":
    {
        "Gradient Descent": [54.48, 43.33, 33.33, 19.64],
        "1": [54.79, 10.36, 1.46, 0.05],
        "2": [54.93, 23.19, 9.44, 1.66],
        "3": [54.50, 24.42, 10.20, 1.98],
        "4": [51.04, 21.31, 8.04, 1.60]
    },
    "Foolbox DeepFool $L_\infty$":
    {
        "Gradient Descent": [54.48, 43.12, 32.51, 17.2],
        "1": [54.79, 9.2, 0.66, 0.01],
        "2": [54.93, 22.46, 7.88, 0.64],
        "3": [54.50, 23.71, 8.57, 0.86],
        "4": [51.04, 20.52, 6.53, 0.64]
    },
    "NES $L_\infty$":
    {
        "Gradient Descent": [54.48, 47.80, 35.40],
        "1": [54.79, 25.23, 3.53],
        "2": [54.93, 35.09, 13.52],
        "3": [54.50, 36.32, 14.10],
        "4": [51.04, 32.51, 11.84]
    },
    "SPSA $L_\infty$":
    {
        "Gradient Descent": [54.48, 45.73, 36.43, 28.13],
        "1": [54.79, 17.77, 3.40, 0.59],
        "2": [54.93, 29.76, 13.73, 5.73],
        "3": [54.50, 31.11, 14.42, 6.18],
        "4": [51.04, 27.36, 12.21, 4.80]
    },
    "NAttack $L_\infty$":
    {
        "Gradient Descent": [54.48, 19.90, 10.54, 4.54],
        "1": [54.79, 39.41, 32.00, 25.96],
        "2": [54.93, 37.53, 38.00, 38.14],
        "3": [54.50, 31.17, 31.98, 32.28],
        "4": [51.04, 39.51, 39.41, 39.13]
    },
    "FGSM CE $L_2$":
    {
        "Gradient Descent": [54.48, 54.48, 37.35, 27.45, 12.00],
        "1": [54.79, 6.22, 6.22, 6.22, 6.22],
        "2": [54.93, 6.26, 6.26, 6.26, 6.26],
        "3": [54.50, 5.92, 5.92, 5.92, 5.92],
        "4": [51.04, 6.85, 6.85, 6.85, 6.85]
    },
    "Newtonfool $L_2$":
    {
        "Gradient Descent": [54.48, 54.5, 54.44, 54.16, 48.59, 37.49, 27.84],
        "1": [54.79, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        "2": [54.93, 10.03, 10.03, 10.03, 10.03, 10.03, 10.03],
        "3": [54.50, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        "4": [51.04, 9.98, 9.98, 9.98, 9.98, 9.98, 9.98]
    },
    "S&P $L_2$":
    {
        "Gradient Descent": [54.48, 54.5, 54.5, 54.16, 51.5, 47.48, 33.28, 22.46],
        "1": [54.79, 54.81, 54.74, 53.73, 44.6, 29.7, 5.57, 1.6],
        "2": [54.93, 54.95, 54.89, 54.23, 47.6, 36.85, 14.4, 6.51],
        "3": [54.50, 54.52, 54.48, 53.89, 47.76, 38.79, 17.02, 8.29],
        "4": [51.04, 51.06, 51.02, 50.52, 44.72, 34.88, 14.63, 6.85]
    },
    "AUN $L_\infty$":
    {
        "Gradient Descent": [54.48, 54.48, 54.47, 54.44, 52.74, 34.54, 17.49, 13.52, 12.42],
        "1": [54.79, 54.75, 54.63, 54.81, 50.93, 26.34, 16.74, 12.61, 11.21],
        "2": [54.93, 54.86, 54.97, 54.97, 53.73, 37.99, 22.81, 13.21, 11.69],
        "3": [54.50, 54.54, 54.44, 54.26, 53.54, 38.29, 24.88, 14.05, 11.59],
        "4": [51.04, 51.11, 51.07, 50.99, 49.28, 34.42, 19.92, 12.25, 11.06]
    }
}

'''
"Gradient Descent": 54.48
"1": 54.79
"2": 54.93
"3": 54.50
"4": 51.04
'''

def scale_values(attack_acc):
    arr = []
    for acc in attack_acc:
        arr.append(((acc)/attack_acc[0])*100)
    return arr


def plot_graph(bottom_zero = False):
    for k, attack in enumerate(data.values()):
        _, ax = plt.subplots(1)
        plt.locator_params(axis='x',nbins=6)
        for i, model in enumerate(attack):
            y = scale_values(attack[model])
            plt.plot(epsilons[k], y, label=model)
        
        plt.xlabel("Epsilon value")
        plt.ylabel("Accuracy loss")
        plt.title(f"{list(data.keys())[k]} accuracy-loss over different perturbation budgets")
        if bottom_zero:
            ax.set_ylim(bottom=0)
        plt.legend()
        filename = (list(data.keys())[k]).replace("\\", "")
        plt.savefig(f"../images/accuracylossplot_{filename}.png")
        plt.clf()
    
plot_graph(True)