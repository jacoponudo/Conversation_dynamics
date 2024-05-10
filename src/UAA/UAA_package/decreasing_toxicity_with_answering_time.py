import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


data['is_toxic']=data['toxicity_score']>0.6
df = data.dropna(subset=['temporal_distance_from_previous_comment_h'])
df=df[df['sequential_number_of_comment_by_user_in_thread']>12]
df['quick_reply']=df['temporal_distance_from_previous_comment_h']<5

# Scatter plot with sensible labels and cute colors
plt.scatter(df['temporal_distance_from_previous_comment_h'], df['toxicity_score'], s=20, alpha=0.05, color='skyblue')

# Label the axes with sensible labels
plt.xlabel('Temporal Distance from Previous Comment (hours)')
plt.ylabel('Toxicity Score')
plt.xlim(0, 100)

# Title of the plot
plt.title('Relationship between Temporal Distance and Toxicity')



# Assuming 'df' is your DataFrame and the columns of interest are 'temporal_distance_from_previous_comment_h' and 'toxicity_score'
x = df['temporal_distance_from_previous_comment_h']
y = df['toxicity_score']

# Calculate Pearson correlation coefficient and p-value
corr_coeff, p_value = pearsonr(x, y)

print("Pearson correlation coefficient:", corr_coeff)
print("p-value:", p_value)

if p_value < 0.05:
    print("There is a statistically significant correlation between the two variables.")
else:
    print("There is no statistically significant correlation between the two variables.")

# Recipe for a simple linear regression model
from sklearn.linear_model import LinearRegression

# Reshape x to a 2D array since scikit-learn expects the input to be 2D
x = x.values.reshape(-1, 1)

# Create and fit the model
model = LinearRegression()
model.fit(x, y)

# Print the coefficients
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

plt.plot(x, model.predict(x), color='red', linewidth=0.5, label=' (Estimated by Model)')

# Show the legend
plt.legend()

# Show the plot
plt.show()