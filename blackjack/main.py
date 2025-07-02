import random
from cards import Card, CardSuit, Hand
from agent import Agent

STEP = 10000
MIN_ITR = 2000
BAR = '█'
FILLER = '▒'


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

    def run_match(self, exploring_starts: bool = False) -> int:
        self.deal()
        ctr = 0
        while True:
            if self.agent.control:
                ctr += 1
            action = self.agent.choose_action(
                self.dealer_hand.show(),
                self.hand,
                ctr == 1 and exploring_starts,
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

    def run_force_match(
        self,
        dealer_show: int,
        player_sum: int,
        usable_ace: int,
        to_do_action: int
    ) -> int:
        self.dealer_hand.add_card(
            Card(
                random.choice([suit for suit in CardSuit]),
                dealer_show
            )
        )
        self.dealer_hand.add_card(self.draw_card())
        if usable_ace == 1:
            self.hand.add_card(
                Card(
                    random.choice([suit for suit in CardSuit]),
                    1
                )
            )
        else:
            self.hand.add_card(
                Card(
                    random.choice([suit for suit in CardSuit]),
                    player_sum // 2
                )
            )
        to_add = player_sum - self.hand.sum()
        if to_add == 11:
            to_add = 1
        self.hand.add_card(
            Card(
                random.choice([suit for suit in CardSuit]),
                to_add
            )
        )
        # now remain the case where there is no usable ace and sum is 21
        if player_sum == 21 and usable_ace == 0:
            self.hand.flush()
            self.hand.add_card(
                Card(
                    random.choice([suit for suit in CardSuit]),
                    10
                )
            )
            self.hand.add_card(
                Card(
                    random.choice([suit for suit in CardSuit]),
                    10
                )
            )
            self.hand.add_card(
                Card(
                    random.choice([suit for suit in CardSuit]),
                    1
                )
            )
        self.agent.SAR_list.append(
            (
                (dealer_show-1, player_sum-12, usable_ace == 1),
                to_do_action,
                0
            )
        )
        ctr = 0
        while True:
            if self.agent.control:
                ctr += 1
            if ctr == 1:
                action = to_do_action
            else:
                action = self.agent.choose_action(
                    self.dealer_hand.show(),
                    self.hand,
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

    def evaluate_policy(self) -> None:
        self.agent = Agent()
        for i in range(STEP*10):
            if i % STEP == 0:
                print(f'Processing iteration {i} for evaluating policy that hits till sum exceeds 19')  # Noqa:E501
            reward = self.run_match()
            self.agent.evaluate_value_function(reward)
            self.agent.reset()
            self.reset()
        self.agent.draw_value_function('policy_eval.png')

    def exploring_starts(self) -> None:
        self.agent = Agent(True)
        print('['+' '*20+']', end='', flush=True)
        for ctr in range(MIN_ITR):
            if ctr % int(MIN_ITR/20) == 0:
                print('\b'*37, end='', flush=True)
                print('[' + (BAR*int(ctr/MIN_ITR * 20)).ljust(20, FILLER) + ']' + f' completed: {int(ctr/MIN_ITR*100):02}%', end='', flush=True)  # Noqa:E501
            for i in range(10):
                for j in range(10):
                    for k in range(2):
                        for m in range(2):
                            reward = self.run_force_match(
                                i+1,
                                j+12,
                                k,
                                m
                            )
                            self.agent.evaluate_value_function(reward)
                            self.agent.reset()
                            self.reset()
        print('\b'*37, end='', flush=True)
        print('['+BAR*20+'] completed: 100%')
        self.agent.draw_policy('exploring_starts.png')


game = BlackJack()
print('Evaluating policy for hitting only at sum greater than 19')
game.evaluate_policy()
print('Starting Exploring starts method')
game.exploring_starts()
