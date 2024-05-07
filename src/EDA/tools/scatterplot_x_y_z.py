import matplotlib.pyplot as plt
import pandas as pd

# Assuming dataset_plot is your DataFrame with columns x, y, and z

# Define color based on 'x' values

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df_plot['x'], df_plot['y'],  alpha=0.005)
plt.title('Scatter Plot with Color Coded by x')
plt.xlabel('numero medio di commenti')
plt.ylabel('quanto sono deep i post con cui interagsce')
plt.show()






