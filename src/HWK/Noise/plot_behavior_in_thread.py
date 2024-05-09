x = [0]*len(ℋ_t)
colors = ['r' if score > 0.5 else 'black' for score in 𝒯_T]
x = [.1 if score > 0.5 else 0 for score in 𝒯_T]
# Plot the scatter plot with colors
plt.scatter(ℋ_t, x, c=colors, label='ℋ_t', alpha=0.1,s=200)

# Plot the horizontal line
plt.axhline(0, color='r', linestyle='--', label='Horizontal Line')

# Add labels and legend
plt.xlabel('Index')
plt.ylabel('Values')
plt.xlim(0, max(ℋ_t) + mean_lag)
plt.legend()

# Show plot
plt.show()