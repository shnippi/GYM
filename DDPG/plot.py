import matplotlib.pyplot as plt
import numpy as np
import os


def simple_plot(scores, mean_scores, epoch):
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    # plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    if epoch % 10 == 0:
        plt.show()


def plot_learning_curve(x, scores, figure_file, algo, env_id):

    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title(f'{algo} {env_id} avg 100 scores')
    plt.savefig(figure_file)
    plt.close()
