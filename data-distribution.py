import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots as sp

plt.style.use('science')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the "abc.txt" file
data = pd.read_csv('MOF_Data.txt', header=None, names=['Density', 'GSA', 'VSA', 'VF', 'PV', 'LCD', 'PLD', 'GC', 'HHH'])

colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'darkgreen', 'gray']

# Create a violin plot for each feature
for i, feature in enumerate(['Density', 'GSA', 'VSA', 'VF', 'PV', 'LCD', 'PLD', 'GC', 'HHH']):
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.histplot(x=data[feature], color=colors[i],kde=True, fill=True, bins=30,edgecolor='black')
    #plt.title(f'Violin Plot for {feature}', fontsize=16,  weight='bold')
    plt.xlabel(feature, fontsize=28, weight='bold')
    plt.ylabel('Frequency', fontsize=28, weight='bold')
    # Make x and y ticks bold
    plt.xticks(fontweight='bold', fontsize=28)
    plt.yticks(fontweight='bold', fontsize=28)
    sns.despine(left=True, bottom=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{feature}_Final.png", dpi=500)
