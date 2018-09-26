import numpy as np
import matplotlib.pyplot as plt


def plot_scores(scores, solved_score=13, title="Title"):
    """
        Plot the scores/rewards per episode
    """
    
    color_scores = "#d58506"
    fig,ax = plt.subplots(figsize=(10,7))
    ax.set_title(title)
    avg_size=100
    nb_episodes = len(scores)
    x_data = np.asarray(range(nb_episodes))-avg_size+1
    ax.plot(x_data, scores, color=color_scores, alpha=0.4, label='Score')

    scores_mov_avg = np.convolve(scores, np.ones((avg_size,)) / avg_size, mode='valid')
    x_data = x_data[avg_size-1:]
    ax.plot(x_data, scores_mov_avg, color=color_scores, label=f'Avg({avg_size}) Score')


    solved_index = np.argmax(scores_mov_avg >= solved_score)
    if solved_index:
        solved_value = scores_mov_avg[solved_index]
        #solved_index += avg_size
        ax.plot(solved_index, solved_value, 'o', markersize=12, color="#3eb116",
                label='Solved Avg Score: %.2f (ep: %d)' % (solved_value, solved_index))

    max_index = np.argmax(scores_mov_avg)
    max_value = scores_mov_avg[max_index]
    #max_index += avg_size
    ax.plot(max_index, max_value, '^', markersize=13, color="#3eb116",
            label='Max Avg Score: %.2f (ep: %d)' % (max_value, max_index))

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Score")

    ax.set_ylim([0, np.max(scores)])
    ax.set_xlim([-avg_size, (len(scores)-avg_size)*1.01])

    ax.legend()


