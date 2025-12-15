import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def sort_scores(history_index, history_value):
    No_D = history_index.shape[1]
    filtering_order = np.zeros(No_D, dtype=int)

    # Determine the filtering order
    filtering_order[0] = np.where(history_index[0] == 0)[0][0]
    for i in range(len(history_index[:, 0]) - 1):
        filtering_order[i + 1] = np.where(history_index[i + 1] - history_index[i] == -1)[0][0]
    filtering_order[-1] = np.where(history_index[-1] == 1)[0][0]

    # Sort the matrix and names based on filtering order
    matrix_sorted = history_value[:, filtering_order]
    return filtering_order, matrix_sorted, 

def plot_filtering_history(history_index, history_value, name_tag, diff_score = False,save_path=None, save_eps=False):
    """
    Plot the filtering history of dictionary items based on their scores.

    Parameters:
    ----------
    history_index : numpy.ndarray
        A 2D array where each row represents the filtering state of dictionary items at a specific step.
        A value of 1 indicates the item is retained, and 0 indicates it is filtered.
    history_value : numpy.ndarray
        A 2D array containing the scores of dictionary items at each filtering step.
    name_tag : list
        A list of names corresponding to the dictionary items.
    save_path : str, optional
        Path to save the plot as a PNG file. Default is None.
    save_eps : bool, optional
        Whether to save the plot in EPS format. Default is False.

    Returns:
    -------
    None
    """

    cmap_colors = [
        '#FFFFFF',  # White
        '#D1E8FF',  # Light Blue
        '#B0D6F3',  # Light Sky Blue
        '#A5CBE6',  # Soft Blue
        # '#C6F2D9',  # Mint Green
        '#A8E4C1',  # Pastel Green
        '#D3F9C9',  # Light Green
        '#FFF9D4',  # Light Yellow
        '#FFD9B3',  # Peach Puff
        '#FFC0CB',  # Light Pink
        '#FF9999',  # Red
    ]
    cmap = ListedColormap(cmap_colors)

    bound = [-1, 0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    bounds = bound + [10]  # Add an upper bound
    norm = BoundaryNorm(bounds, len(cmap_colors))

    filtering_order, matrix_sorted = sort_scores(history_index, history_value)
    if diff_score:
        bound = [-1, 0, 1e-2, 5e-2, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3]
        bounds = bound + [1e5]  # Add an upper bound
        norm = BoundaryNorm(bounds, len(cmap_colors))
        new_sorted = matrix_sorted.copy()
        for i in range(matrix_sorted.shape[0]-1):
            new_sorted[i+1, :] = matrix_sorted[i+1, :] / matrix_sorted[i, i]
        matrix_sorted = new_sorted
    name_tag_sorted = [name_tag[i] for i in filtering_order]

    # Plot the filtering history
    plt.figure(figsize=(30, 6))
    im = plt.imshow(matrix_sorted, cmap=cmap, norm=norm, aspect=0.3)

    # Add horizontal lines for filtering steps
    for y in range(1, len(history_index[:, 0])):
        plt.axhline(y=y - 0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Add text labels for dictionary items
    for i in range(len(matrix_sorted[0, :]) - 1):
        plt.text(i-0.5, i, name_tag_sorted[i], ha='left', va='center')
    plt.text(i + 0.5, i, name_tag_sorted[i + 1], ha='left', va='center')

    # Add colorbar
    cbar = plt.colorbar(im, ticks=bound, label="Score", shrink=0.8, aspect=18)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("Score", size=18)

    # Add titles and labels
    plt.title("Score history (White: filtered)", fontsize=20)
    plt.xticks(ticks=np.arange(len(name_tag_sorted)), labels=name_tag_sorted, fontsize=21, rotation=90)
    plt.xlabel("Dictionary items", fontsize=18)
    plt.ylabel("Filtering order (0, 1, 2, 3, ...)", fontsize=18)

    # Save the plot if a path is provided
    if save_path is not None:
        if save_eps:
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
        else:
            plt.savefig(save_path, bbox_inches='tight')

def plot_filtering_history_sorted(filtering_order, history_value, name_tag, backwards = True, diff_score = False,save_path=None, save_eps=False, print_inside = True):
    """
    Plot the filtering history of dictionary items based on their scores.

    Parameters:
    ----------
    history_index : numpy.ndarray
        A 2D array where each row represents the filtering state of dictionary items at a specific step.
        A value of 1 indicates the item is retained, and 0 indicates it is filtered.
    history_value : numpy.ndarray
        A 2D array containing the scores of dictionary items at each filtering step.
    name_tag : list
        A list of names corresponding to the dictionary items.
    save_path : str, optional
        Path to save the plot as a PNG file. Default is None.
    save_eps : bool, optional
        Whether to save the plot in EPS format. Default is False.

    Returns:
    -------
    None
    """

    cmap_colors = [
        '#FFFFFF',  # White
        '#D1E8FF',  # Light Blue
        '#B0D6F3',  # Light Sky Blue
        '#A5CBE6',  # Soft Blue
        # '#C6F2D9',  # Mint Green
        '#A8E4C1',  # Pastel Green
        '#D3F9C9',  # Light Green
        '#FFF9D4',  # Light Yellow
        '#FFD9B3',  # Peach Puff
        '#FFC0CB',  # Light Pink
        '#FF9999',  # Red
    ]
    cmap = ListedColormap(cmap_colors)

    bound = [-1, 0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    bounds = bound + [10]  # Add an upper bound
    norm = BoundaryNorm(bounds, len(cmap_colors))

    matrix_sorted = history_value[:, filtering_order]
    if diff_score:
        bound = [-1, 0, 1e-2, 5e-2, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3]
        bounds = bound + [1e5]  # Add an upper bound
        norm = BoundaryNorm(bounds, len(cmap_colors))
        new_sorted = matrix_sorted.copy()
        for i in range(matrix_sorted.shape[0]-1):
            new_sorted[i+1, :] = matrix_sorted[i+1, :] / matrix_sorted[i, i]
        matrix_sorted = new_sorted
    name_tag_sorted = [name_tag[i] for i in filtering_order]
    name_tag_sorted = [nt.replace('^1','') if '^1' in nt else nt for nt in name_tag_sorted]
    name_tag_sorted = [f'${nt}$' for nt in name_tag_sorted]

    # Plot the filtering history
    plt.figure(figsize=(30, 6))
    im = plt.imshow(matrix_sorted, cmap=cmap, norm=norm, aspect=0.3)

    # Add horizontal lines for filtering steps
    print(matrix_sorted.shape)
    for y in range(1, matrix_sorted.shape[0]):
        plt.axhline(y=y - 0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Add text labels for dictionary items
    if print_inside:
        if backwards:
            for i in range(matrix_sorted.shape[0]):
                for j in range(i,matrix_sorted.shape[1]):
                    plt.text(j-0.2, i, f'{matrix_sorted[i,j]:0.3g}', ha='left', va='center')
            pos = i + 0.5
            # j+=1
            # for nmtg in name_tag_sorted[i+1:]:
            #     plt.text(pos, i,  f'{matrix_sorted[i,j]:0.1g}', ha='left', va='center')
            # pos+=1
        else:
            for i in range(matrix_sorted.shape[0]):
                plt.text(i-0.5, matrix_sorted.shape[0] - i-1, name_tag_sorted[i], ha='left', va='center')
            pos = i + 0.5
            for nmtg in name_tag_sorted[i+1:]:
                plt.text(pos, matrix_sorted.shape[0] - i-1, nmtg, ha='left', va='center')
            pos+=1


    # Add colorbar
    cbar = plt.colorbar(im, ticks=bound, label="Score", shrink=0.8, aspect=18)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("Score", size=18)

    # Add titles and labels
    plt.title("Score history (White: filtered)", fontsize=20)
    plt.xticks(ticks=np.arange(len(name_tag_sorted)), labels=name_tag_sorted, fontsize=32, rotation=90)
    plt.xlabel("Dictionary items", fontsize=18)
    plt.ylabel("Filtering order (0, 1, 2, 3, ...)", fontsize=18)

    # Save the plot if a path is provided
    if save_path is not None:
        if save_eps:
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
        else:
            plt.savefig(save_path, bbox_inches='tight')


def non_uniform_diff(X,t):
    h0 = (t[1:-1] - t[:-2])[:, None]
    h1 = (t[2:] - t[1:-1])[:, None]
    x_dot = np.zeros_like(X)
    x_dot[0] = (X[1] - X[0]) / h0[0]
    x_dot[-1] = (X[-1] - X[-2]) / h1[-1]
    x_dot[1:-1] = (h1 - h0) / (h0 * h1) * X[1:-1] + (1 / (h0 + h1)) * ((h0 / h1) * X[2:] - (h1 / h0) * X[:-2])
    return x_dot

def create_lambda_functions(expressions, inputs=['x', 'y', 'w']):
    """
    Create a list of lambda functions from a list of mathematical expressions.

    Parameters:
    ----------
    expressions : list of str
        List of mathematical expressions as strings.

    Returns:
    -------
    list of lambda functions
        Each lambda function corresponds to an expression in the input list.
    """
    expressions = [x.replace('^', '**') for x in expressions]
    expressions = [x.replace(' ', '*') for x in expressions]
    expressions = [x.replace('sin', 'np.sin') for x in expressions]
    expressions = [x.replace('cos', 'np.cos') for x in expressions]

    expressions_names = [x.replace('**', '+"^"+') for x in expressions]
    expressions_names = [x.replace('*', '+ " " +') for x in expressions_names]
    expressions_names = [re.sub(r'(?<!\w)(\d+)(?!\w)', r"'\1'", x) for x in expressions_names]

    print(expressions_names)
    lambda_functions = []
    lambda_functions_names = []
    for expr, expr_nm in zip(expressions, expressions_names):
        # Create a lambda function for each expression
        func = eval(f"lambda {','.join(inputs)}: {expr}")
        func_name = eval(f"lambda {','.join(inputs)}: {expr_nm}")
        lambda_functions.append(func)
        lambda_functions_names.append(func_name)
    return lambda_functions, lambda_functions_names