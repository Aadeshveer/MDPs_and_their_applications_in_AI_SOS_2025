import numpy as np
import random
from cards import Card, CardSuit, Hand
import matplotlib.pyplot as plt


class Agent:

    def __init__(self) -> None:
        # 10 values for dealer card, 18(4-21) for self sum, 1 for ace
        self.values = np.zeros((10, 18, 2))
        self.state_ctr = np.zeros((10, 18, 2), dtype=int)
        # Choose to hit for idx 1
        self.policy = np.zeros((10, 18, 2, 2), dtype=np.int8)
        self.policy[:, 16:, :, 0] = 1
        self.policy[:, :16, :, 1] = 1
        # SAR = [(state(dealer, sum, ace), action, reward),...]
        self.SAR_list: list[tuple[tuple[int, int, bool], int, int]] = []
        self.GAMMA = 1

    def choose_action(self, dealer_card: Card, hand: Hand) -> int:
        sum = hand.sum()
        dealer_show = min(dealer_card.value, 10)
        usable_ace = hand.usable_ace()

        if self.policy[
            dealer_show-1,
            sum-4,
            int(usable_ace),
            1
        ] > random.random():
            action = 1
        else:
            action = 0
        self.SAR_list.append(((dealer_show-1, sum-4, usable_ace), action, 0))
        return action

    def evaluate_value_function(self, final_reward: int):
        self.SAR_list.reverse()
        final_state, final_action, _ = self.SAR_list[0]
        self.SAR_list[0] = (final_state, final_action, final_reward)

        g = 0
        for state, _action, reward in self.SAR_list:
            self.state_ctr[*state] += 1
            g = self.GAMMA * g + reward
            state_loc = (
                state[0],
                state[1],
                int(state[2]),
            )
            V_k = self.values[*state_loc]
            if self.state_ctr[*state_loc] == 0:
                self.state_ctr[*state_loc] = 1
                self.values[*state_loc] = g
            else:
                self.values[*state_loc] = V_k
                self.values[*state_loc] += (g - V_k)/self.state_ctr[*state_loc]

    def draw(self):
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle("State-Value Function", fontsize=16)

        values = self.values[:, 8:, :]

        h, w, _ = values.shape

        player_sum_range = np.arange(12, 12 + w)  # e.g., [12, 13, ..., 21]
        dealer_card_range = np.arange(1, 1 + h)   # e.g., [1, 2, ..., 10]

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
        ax2.set_zticks(np.array([0, 1]))
        fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=10, label="State Value")
        ax2.view_init(elev=30, azim=-135) 

        plt.tight_layout()
        plt.savefig('policy_eval.png')

    def reset(self):
        self.SAR_list = []


class BlackJack:

    def __init__(self, n_decks: int = 0) -> None:
        # note that 0 n_decks means any card can be drawn
        self.n_decks = n_decks
        # deck is normal deck of cards
        deck: list[Card] = []
        for value in range(1, 14):
            for suit in CardSuit:
                deck.append(Card(suit, value))

        self.play_deck: list[Card] = []
        for _ in range(n_decks):
            self.play_deck.extend(deck)

        self.hand = Hand()
        self.dealer_hand = Hand()

        self.agent = Agent()

    def draw_card(self) -> Card:
        if len(self.play_deck) == 0:
            return Card(
                random.choice([suit for suit in CardSuit]),
                random.randint(1, 13)
            )
        else:
            card = self.play_deck.pop()
            return card

    def deal(self) -> None:
        random.shuffle(self.play_deck)
        self.dealer_hand.add_card(self.draw_card())
        self.dealer_hand.add_card(self.draw_card())
        self.hand.add_card(self.draw_card())
        self.hand.add_card(self.draw_card())

    def hit(self) -> None:
        self.hand.add_card(self.draw_card())

    def stand(self) -> None:
        while self.dealer_hand.sum() < 17:
            self.dealer_hand.add_card(self.draw_card())

    def reset(self) -> None:
        if self.n_decks > 0:
            self.play_deck.extend(self.hand.flush())
            self.play_deck.extend(self.dealer_hand.flush())
        else:
            self.hand.flush()
            self.dealer_hand.flush()

    def run_match(self) -> int:
        self.deal()
        while True:
            action = self.agent.choose_action(
                self.dealer_hand.show(),
                self.hand
            )
            if action == 1:
                self.hit()
                if self.hand.sum() > 21:
                    return -1
            else:
                self.stand()
                break
        if self.dealer_hand.sum() > 21:
            return 1
        elif self.dealer_hand.sum() > self.hand.sum():
            return -1
        elif self.dealer_hand.sum() < self.hand.sum():
            return 1
        else:
            return 0

    def iterate(self):
        for i in range(500000):
            if i % 10000 == 0:
                print(f'Processing iteration {i}')
            reward = self.run_match()
            self.agent.evaluate_value_function(reward)
            self.agent.reset()
            self.reset()
        self.agent.draw()


game = BlackJack()
game.iterate()
