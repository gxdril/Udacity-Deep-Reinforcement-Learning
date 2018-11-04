import numpy as np
import matplotlib.pyplot as plt


def plot_scores(scores, solved_score=0.5, title="Score/episode"):
    """
        Plot the scores/rewards per episode
    """
    
    color_scores = "#d58506"
    fig,ax = plt.subplots(figsize=(10,7))
    ax.set_title(title)
    avg_size=100
    nb_episodes = len(scores)
    x_data = np.asarray(range(nb_episodes))+1
    ax.plot(x_data, scores, color=color_scores, alpha=0.4, label='Score')

    scores_mov_avg = np.convolve(scores, np.ones((avg_size,)) / avg_size, mode='valid')
    x_data = x_data[avg_size-1:]
    ax.plot(x_data, scores_mov_avg, color=color_scores, label=f'Avg({avg_size}) Score')
    ax.axhline (solved_score,color='b',linestyle=":", alpha=0.3)
    if np.min(scores)<0:
        ax.axhline (0,color='b',linestyle="-", alpha=0.2)
    solved_index = np.argmax(scores_mov_avg >= solved_score)

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Score")

    ax.set_ylim([np.min(scores)*1.01 if np.min(scores)<0 else 0, np.max(scores)*1.01])
    ax.set_xlim([0, (len(scores))*1.01])

    ax.legend()


