
plt.boxplot([z[z.index.isin(ids_deep_chat)], z[z.index.isin(ids_flash_conversations)]],showfliers=False)

plt.title('Titolo')
plt.ylabel('Valori')
plt.xlabel('Gruppi')
plt.grid(True)
plt.show()
z[z.index.isin(ids_deep_chat)].median()
z[z.index.isin(ids_flash_conversations)].median()
