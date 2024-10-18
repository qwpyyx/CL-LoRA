import matplotlib.pyplot as plt

# Updating the x-axis to only show the provided ranks without continuous values
# ranks = [16, 32, 64, 128]
# olora_scores = [76.04, 76.09, 73.1, 65.63]
#
# # Plotting the data
# plt.figure(figsize=(8, 5))
# plt.plot(ranks, olora_scores, marker='o', label='O-LoRA', linestyle='--', color='green')
#
# # Titles and labels
# plt.title('IncLoRA and O-LoRA in Order 1', fontsize=12)
# plt.xlabel('Ranks', fontsize=12)
# plt.ylabel('Exact Match Score', fontsize=12)
#
# # Adding grid, legend and improving visualization
# plt.grid(True)
# plt.legend(title='Algorithm', fontsize=10)
# plt.xticks(ranks, fontsize=10)  # Ensuring only the specified ranks are shown on x-axis
# plt.yticks(fontsize=10)
# plt.ylim([0, 100])
#
# # Show the plot
# plt.tight_layout()
# plt.show()




# Data from the table
# tasks = ['dbpedia', 'amazon', 'agnews', 'yahoo']
# icrlora_scores = [98.62,	94.50,	72.95,	20.26]
# olora_scores = [98.62,	98.45,	97.64,	97.37]
#
# # Plotting the data
# plt.figure(figsize=(8, 5))
# plt.plot(tasks, icrlora_scores, marker='o', label='IncLoRA', linestyle='-', color='blue')
# plt.plot(tasks, olora_scores, marker='o', label='O-LoRA', linestyle='--', color='green')
#
# # Titles and labels
# plt.title('IncLoRA and O-LoRA on dbpedia task over multiple tasks in Order 2', fontsize=12)
# plt.xlabel('Tasks', fontsize=12)
# plt.ylabel('Exact Match Score', fontsize=12)
#
# # Adding grid, legend and improving visualization
# plt.grid(True)
# plt.legend(title='Algorithm', fontsize=10)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.ylim([0, 100])
#
# # Show the plot
# plt.tight_layout()
# plt.show()



# # Data from the table
# tasks = ['yahoo', 'amazon', 'agnews', 'dbpedia']
# icrlora_scores = [71.01,	70.46,	62.12,	66.12]
# olora_scores = [71.01,	70.89,	68.07,	68.32]
#
# # Plotting the data
# plt.figure(figsize=(8, 5))
# plt.plot(tasks, icrlora_scores, marker='o', label='IncLoRA', linestyle='-', color='blue')
# plt.plot(tasks, olora_scores, marker='o', label='O-LoRA', linestyle='--', color='green')
#
# # Titles and labels
# plt.title('LLaMA IncLoRA and O-LoRA on yahoo task over multiple tasks in Order 3', fontsize=12)
# plt.xlabel('Tasks', fontsize=12)
# plt.ylabel('Exact Match Score', fontsize=12)
#
# # Adding grid, legend and improving visualization
# plt.grid(True)
# plt.legend(title='Algorithm', fontsize=10)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.ylim([0, 100])
#
# # Show the plot
# plt.tight_layout()
# plt.show()

# Data from the table
tasks = ['dbpedia', 'amazon', 'yahoo', 'agnews']
ranks_16 = [98.6184, 98.5658, 98.3289, 98.0789]
ranks_32 = [98.4737, 98.3421, 97.8684, 97.4211]
ranks_64 = [98.6184, 98.6579, 98.5132, 98.5]
ranks_128 = [98.5789, 98.4868, 97.9737, 97.7895]

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(tasks, ranks_16, marker='o', label='Rank 16', linestyle='-', color='blue')
plt.plot(tasks, ranks_32, marker='o', label='Rank 32', linestyle='--', color='green')
plt.plot(tasks, ranks_64, marker='o', label='Rank 64', linestyle='-.', color='red')
plt.plot(tasks, ranks_128, marker='o', label='Rank 128', linestyle=':', color='purple')

# Titles and labels
plt.title('Performance of dbpedia on Different Tasks with Various Ranks in Order 1', fontsize=12)
plt.xlabel('Tasks', fontsize=12)
plt.ylabel('Exact Match Score', fontsize=12)

# Adding grid, legend and improving visualization
plt.grid(True)
plt.legend(title='Ranks', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylim([97, 99])  # Adjusting the y-axis range for better visualization

# Show the plot
plt.tight_layout()
plt.show()
