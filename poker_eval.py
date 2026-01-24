import enum
from collections.abc import Sequence
from itertools import combinations

class HandRank(enum.IntEnum):
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9

class HandEvaluator:
    # Complete mapping for standard ranks (2-14)
    RANK_MAP = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
        '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }

    @staticmethod
    def evaluate_hand(cards: Sequence[str]) -> tuple[HandRank, list[int]]:
        """Evaluate best 5-card hand from any number of cards."""
        if len(cards) < 5:
            # Handle pre-flop/flop cases gracefully if needed, 
            # or just return High Card of what we have.
            # For this agent, we usually assume 5+ cards or handle pre-flop separately.
            return HandRank.HIGH_CARD, []

        best_rank = HandRank.HIGH_CARD
        best_tiebreakers = []

        for combo in combinations(cards, 5):
            rank, tiebreakers = HandEvaluator._evaluate_five_cards(list(combo))
            # Tuple comparison works automatically for (Rank, Tiebreakers)
            if (rank > best_rank) or (rank == best_rank and tiebreakers > best_tiebreakers):
                best_rank = rank
                best_tiebreakers = tiebreakers

        return best_rank, best_tiebreakers

    @staticmethod
    def _evaluate_five_cards(cards: list[str]) -> tuple[HandRank, list[int]]:
        # FIX 1: Sort by RANK VALUE, not string character
        # cards are like "C7", "DT". We need the char at index 1.
        sorted_cards = sorted(cards, key=lambda c: HandEvaluator.RANK_MAP[c[1]], reverse=True)
        
        ranks = [HandEvaluator.RANK_MAP[c[1]] for c in sorted_cards]
        suits = [c[0] for c in sorted_cards]

        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1

        # Sort counts (frequency desc, then rank desc)
        # keys=lambda r: (count, rank_value)
        unique_ranks = sorted(rank_counts.keys(), key=lambda r: (rank_counts[r], r), reverse=True)
        counts = sorted(rank_counts.values(), reverse=True)

        is_flush = len(set(suits)) == 1
        
        # Check Straight
        is_straight = False
        straight_high = ranks[0]
        
        if len(set(ranks)) == 5:
            if ranks[0] - ranks[4] == 4:
                is_straight = True
            # FIX 2: Correct Wheel Straight (A-5-4-3-2) check for 2-14 ranking
            elif ranks == [14, 5, 4, 3, 2]:
                is_straight = True
                straight_high = 5 # 5-high straight

        # 1. Royal / Straight Flush
        if is_straight and is_flush:
            if straight_high == 14 and ranks[1] == 13: # Ensure it's not a wheel
                return HandRank.ROYAL_FLUSH, []
            return HandRank.STRAIGHT_FLUSH, [straight_high]

        # 2. Four of a Kind
        if counts == [4, 1]:
            return HandRank.FOUR_OF_A_KIND, [unique_ranks[0], unique_ranks[1]]

        # 3. Full House
        if counts == [3, 2]:
            return HandRank.FULL_HOUSE, [unique_ranks[0], unique_ranks[1]]

        # 4. Flush
        if is_flush:
            return HandRank.FLUSH, ranks

        # 5. Straight
        if is_straight:
            return HandRank.STRAIGHT, [straight_high]

        # 6. Three of a Kind
        if counts == [3, 1, 1]:
            return HandRank.THREE_OF_A_KIND, [unique_ranks[0]] + sorted(unique_ranks[1:], reverse=True)

        # 7. Two Pair
        if counts == [2, 2, 1]:
            return HandRank.TWO_PAIR, sorted(unique_ranks[:2], reverse=True) + [unique_ranks[2]]

        # 8. Pair
        if counts == [2, 1, 1, 1]:
            return HandRank.PAIR, [unique_ranks[0]] + sorted(unique_ranks[1:], reverse=True)

        # 9. High Card
        return HandRank.HIGH_CARD, ranks