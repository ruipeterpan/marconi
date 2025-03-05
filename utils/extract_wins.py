# %%
import re

def extract_better_configuration_stats(file_path):
    # Open the file and read its content
    with open(file_path, 'r') as file:
        text = file.read()

    # Regular expression pattern to match the values of token_hit_rate_win and flops_savings_win
    pattern = r"Better configuration! token_hit_rate_win ([\d.]+)%, flops_savings_win ([\d.]+)%"
    
    # Find all matches in the text
    matches = re.findall(pattern, text)
    
    # Convert the extracted values into a list of tuples with floats
    return [(float(token_hit), float(flops_savings)) for token_hit, flops_savings in matches]

# Call the function and print the results
better_configs = extract_better_configuration_stats('../lmsys.txt')
print(better_configs)

# %%
import matplotlib.pyplot as plt

def plot_histogram(data, bins=50, title='Histogram', xlabel='Value', ylabel='Frequency'):
    # Create the histogram
    plt.hist(data, bins=bins, edgecolor='black')
    
    # Add title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Show the plot
    plt.show()
# %%
plot_histogram([c[0] for c in better_configs], title='Token hit rate improvement distribution for winning configs', xlabel='Token hit rate absolute improvement (%)', ylabel='Count')
plot_histogram([c[1] for c in better_configs], title='FLOPs savings improvement distribution', xlabel='FLOPs savings improvement (%)', ylabel='Count')

# %%
