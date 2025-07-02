import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_policy_step(ax, policy_slice):
    cmap = matplotlib.cm.get_cmap('viridis')
    im = ax.imshow(np.rot90(policy_slice), cmap=cmap)
    colors = [
        im.cmap(im.norm(0)),
        im.cmap(im.norm(1)),
    ]
    patch = [
        Patch(color=colors[0], label='Stand'),
        Patch(color=colors[1], label='Hit')
    ]
    ax.set_yticks(np.arange(0, 10))
    ax.set_yticklabels(np.arange(21, 11, -1))
    ax.set_xticks(np.arange(0, 10))
    ax.set_xticklabels(['A', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    ax.set_ylabel('Player sum', fontsize=12)
    ax.set_xlabel('Dealer showing', fontsize=12)
    ax.legend(handles=patch)


def plot_value_3d(ax, value_slice):
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)
    X, Y = np.meshgrid(dealer_cards, player_sums)

    Z = value_slice.T

    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Dealer showing', labelpad=10, fontsize=12)
    ax.set_ylabel('Player sum', labelpad=10, fontsize=12)
    ax.set_xticks([1, 10])
    ax.set_xticklabels(['A', '10'])
    ax.set_yticks([12, 21])
    ax.set_zticks([-1, 0, 1])
    ax.tick_params(axis='z', labelsize=12)
    ax.view_init(elev=30, azim=-75)


def draw_policy_and_value(title: str, policy, state_action_values) -> None:

    optimal_policy = np.argmax(policy, axis=3)
    optimal_values = np.max(state_action_values, axis=3)

    fig = plt.figure(figsize=(12, 11))

    ax_pi_usable = fig.add_subplot(2, 2, 1)
    ax_v_usable = fig.add_subplot(2, 2, 2, projection='3d')
    ax_pi_no_usable = fig.add_subplot(2, 2, 3)
    ax_v_no_usable = fig.add_subplot(2, 2, 4, projection='3d')

    plot_policy_step(ax_pi_usable, optimal_policy[:, :, 1])
    ax_pi_usable.set_title('Optimal policy', fontsize=20, pad=20)
    plot_value_3d(ax_v_usable, optimal_values[:, :, 1])
    ax_v_usable.set_title('Optimal value function', fontsize=20, pad=20)

    plot_policy_step(ax_pi_no_usable, optimal_policy[:, :, 0])
    plot_value_3d(ax_v_no_usable, optimal_values[:, :, 0])

    fig.text(0.06, 0.7, 'Usable\n  ace', va='center', ha='center', fontsize=14)
    fig.text(0.06, 0.3, '  No\nusable\n  ace', va='center', ha='center', fontsize=14)  # Noqa:E501

    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.1)
    if title:
        plt.savefig(title)
    else:
        plt.show()


def plot_value_function(title: str, values):
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("State-Value Function", fontsize=16)

    h, w, _ = values.shape

    player_sum_range = np.arange(12, 12 + w)
    dealer_card_range = np.arange(1, 1 + h)

    X, Y = np.meshgrid(player_sum_range, dealer_card_range)

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, values[:, :, 0], cmap='viridis')

    ax1.set_title('No Usable Ace')
    ax1.set_xlabel('Player Sum')
    ax1.set_ylabel('Dealer Showing')
    ax1.set_zlabel('Value')

    ax1.set_xticks(player_sum_range)
    ax1.set_yticks(dealer_card_range)
    ax1.set_zticks(np.array([0, 1]))
    fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=10, label="State Value")
    ax1.view_init(elev=30, azim=-135)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, values[:, :, 1], cmap='plasma')

    ax2.set_title('Usable Ace')
    ax2.set_xlabel('Player Sum')
    ax2.set_ylabel('Dealer Showing')
    ax2.set_zlabel('Value')

    ax2.set_xticks(player_sum_range)
    ax2.set_yticks(dealer_card_range)
    ax2.set_zticks(np.array([-1, 0, 1]))
    fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=10, label="State Value")
    ax2.view_init(elev=30, azim=-135)

    plt.tight_layout()
    if title == '':
        plt.show()
    else:
        plt.savefig(title)
