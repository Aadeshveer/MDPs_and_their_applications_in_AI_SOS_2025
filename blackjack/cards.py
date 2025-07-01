import enum

VALUE_SYMBOL_MAP = {
    1: 'A',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: '10',
    11: 'J',
    12: 'Q',
    13: 'K',
}


class CardSuit(enum.Enum):
    SPADES = 0
    HEARTS = 1
    CLUBS = 2
    DIAMONDS = 3


class Card:

    def __init__(self, suit: CardSuit, value: int) -> None:
        self.suit: CardSuit = suit
        self.value: int = value
        self.symbol: str = VALUE_SYMBOL_MAP[self.value]

    def __str__(self) -> str:
        if len(self.symbol) == 1:
            spacer = ' '
        else:
            spacer = ''
        match self.suit:
            case CardSuit.SPADES:
                symb = '\u2660'
            case CardSuit.HEARTS:
                symb = '\u2665'
            case CardSuit.CLUBS:
                symb = '\u2663'
            case CardSuit.DIAMONDS:
                symb = '\u2666'
        s = ' ___ \n'
        s += '|'+self.symbol+spacer+' |\n'
        s += '| ' + symb + ' |\n'
        s += '|_' + len(spacer)*'_' + self.symbol + '|\n'
        return s


class Hand:

    def __init__(self) -> None:
        self.cards: list[Card] = []

    def add_card(self, card: Card) -> None:
        self.cards.append(card)

    def flush(self) -> list[Card]:
        cards = self.cards
        self.cards = []
        return cards

    def sum(self) -> int:
        ace_ctr: int = 0
        sum: int = 0
        for i in self.cards:
            if i.value == 1:
                ace_ctr += 1
            sum += min(10, i.value)
        while ace_ctr > 0 and sum+10 <= 21:
            sum += 10
            ace_ctr -= 1
        return sum

    def usable_ace(self) -> bool:
        sum: int = 0
        for card in self.cards:
            sum += card.value
        return self.sum() > sum

    def show(self) -> Card:
        return self.cards[0]

    def __str__(self) -> str:
        s = ['', '', '', '']
        for card in self.cards:
            temp = str(card).split('\n')
            for i in range(4):
                s[i] += temp[i] + '   '
        for i in range(4):
            s[i] += '\n'
        return ''.join(s)
