"""
NATHAN Bot - Improved balanced strategy

This bot uses a simple but effective strategy:
- Preflop: Play premium pairs and strong broadway hands aggressively; widen range in late position.
- Postflop: Use the hand evaluator to play strong made hands aggressively, call reasonable bets with top pair,
  and semi-bluff strong draws. Uses pot-based sizing and respects legal actions.
"""
from typing import List, Dict, Any

from bot_api import PokerBotAPI, PlayerAction, GameInfoAPI
from engine.cards import Card, Rank, HandEvaluator
from engine.poker_game import GameState


class NATHANBot(PokerBotAPI):
	def __init__(self, name: str):
		super().__init__(name)
		self.hands_played = 0
		self.hands_won = 0
		self.aggression = 0.6  # tuning parameter for raises
		self.last_raised_preflop = False
		self.min_confidence_to_raise = 0.75
		self.min_confidence_to_play = 0.45
		self.opponent_stats: Dict[str, Dict[str, int]] = {}

	def get_action(self, game_state: GameState, hole_cards: List[Card], 
				   legal_actions: List[PlayerAction], min_bet: int, max_bet: int) -> tuple:
		"""Main decision entry point."""
		# Defensive: if no legal actions, fold
		if not legal_actions:
			return PlayerAction.FOLD, 0

		# Preflop vs postflop branching
		if game_state.round_name == "preflop":
			action = self._preflop(game_state, hole_cards, legal_actions, min_bet, max_bet)
			# remember if we raised preflop to use as a continuation-bet signal
			if action[0] == PlayerAction.RAISE:
				self.last_raised_preflop = True
			else:
				self.last_raised_preflop = False
			return action
		else:
			return self._postflop(game_state, hole_cards, legal_actions, min_bet, max_bet)

	def _preflop(self, game_state: GameState, hole_cards: List[Card], legal_actions: List[PlayerAction],
				 min_bet: int, max_bet: int) -> tuple:
		# Basic sanity
		if len(hole_cards) != 2:
			return PlayerAction.FOLD, 0

		# Evaluate preflop hand strength as a normalized confidence (0..1)
		confidence = self._evaluate_preflop_strength(hole_cards)

		pos = GameInfoAPI.get_position_info(game_state, self.name)
		late = pos.get('is_last', False) or pos.get('players_after', 0) <= 1

		to_call = GameInfoAPI.calculate_bet_amount(game_state.current_bet, game_state.player_bets.get(self.name, 0))

		# Strong hands: raise aggressively
		if confidence >= self.min_confidence_to_raise:
			if PlayerAction.RAISE in legal_actions:
				amount = self._choose_raise_amount(game_state, min_bet, max_bet, factor=2.5 if late else 2.0)
				return PlayerAction.RAISE, amount
			if PlayerAction.CALL in legal_actions and to_call <= game_state.big_blind * 2:
				return PlayerAction.CALL, 0

		# Medium strength: play in late position or call small bets
		if confidence >= self.min_confidence_to_play:
			if to_call == 0:
				# If it's a free option, limp/check
				if PlayerAction.CHECK in legal_actions:
					return PlayerAction.CHECK, 0
				# otherwise raise small in late position
				if late and PlayerAction.RAISE in legal_actions:
					return PlayerAction.RAISE, self._choose_raise_amount(game_state, min_bet, max_bet, factor=1.5)
			else:
				# Facing a bet - call only if pot odds reasonable
				pot_odds = GameInfoAPI.get_pot_odds(game_state.pot, to_call)
				if pot_odds >= 2.5 and PlayerAction.CALL in legal_actions:
					return PlayerAction.CALL, 0

		# Otherwise fold or check
		if PlayerAction.CHECK in legal_actions:
			return PlayerAction.CHECK, 0
		return PlayerAction.FOLD, 0

	def _evaluate_preflop_strength(self, hole_cards: List[Card]) -> float:
		"""Return a confidence 0.0..1.0 for preflop hand strength.
		Simple heuristic: pocket pairs, high-card combos, suitedness, connectedness."""
		c1, c2 = hole_cards
		r1, r2 = c1.rank.value, c2.rank.value
		suited = c1.suit == c2.suit
		high_card_bonus = (max(r1, r2) >= Rank.KING.value) * 0.18 + (max(r1, r2) == Rank.ACE.value) * 0.12
		pair_bonus = 0.3 if r1 == r2 else 0.0
		distance = abs(r1 - r2)
		connector_bonus = 0.15 if distance <= 1 else (0.08 if distance == 2 else 0.0)
		suited_bonus = 0.12 if suited else 0.0
		base = (r1 + r2) / (2.0 * Rank.ACE.value)
		score = base * 0.4 + pair_bonus + connector_bonus + suited_bonus + high_card_bonus
		return min(1.0, score)

	def _postflop(self, game_state: GameState, hole_cards: List[Card], legal_actions: List[PlayerAction],
				  min_bet: int, max_bet: int) -> tuple:
		# Evaluate current hand strength
		all_cards = hole_cards + game_state.community_cards
		hand_type, _, best5 = HandEvaluator.evaluate_best_hand(all_cards)
		rank_value = HandEvaluator.HAND_RANKINGS[hand_type]

		# Quick helpers for legal action fallbacks
		def fallback_check_call_fold():
			if PlayerAction.CHECK in legal_actions:
				return PlayerAction.CHECK, 0
			if PlayerAction.CALL in legal_actions:
				return PlayerAction.CALL, 0
			return PlayerAction.FOLD, 0

		# If we raised preflop and nobody has bet yet, consider a continuation bet
		if self.last_raised_preflop and game_state.current_bet == 0 and PlayerAction.RAISE in legal_actions:
			# only c-bet with at least a pair or a strong draw
			if rank_value >= HandEvaluator.HAND_RANKINGS.get('pair', 1) or self._has_strong_draw(all_cards):
				return PlayerAction.RAISE, self._choose_raise_amount(game_state, min_bet, max_bet, factor=0.6)

		# Strong made hands (two pair or better): be aggressive
		if rank_value >= HandEvaluator.HAND_RANKINGS.get('two_pair', 2):
			if PlayerAction.RAISE in legal_actions:
				return PlayerAction.RAISE, self._choose_raise_amount(game_state, min_bet, max_bet, factor=1)
			return fallback_check_call_fold()

		# Top pair / pair: defend reasonably
		if rank_value >= HandEvaluator.HAND_RANKINGS.get('pair', 1):
			# If pot is large relative to bet, call; if we can raise to protect, do so sometimes
			pot = game_state.pot
			if PlayerAction.CALL in legal_actions:
				to_call = GameInfoAPI.calculate_bet_amount(game_state.current_bet, game_state.player_bets[self.name])
				pot_odds = GameInfoAPI.get_pot_odds(pot, to_call)
				if pot_odds >= 1.5 or to_call == 0:
					# Good odds or free, call
					# Occasionally raise to extract value
					if PlayerAction.RAISE in legal_actions and self._should_bluff_or_value(pot, rank_value):
						return PlayerAction.RAISE, self._choose_raise_amount(game_state, min_bet, max_bet, factor=1)
					return PlayerAction.CALL, 0
			return fallback_check_call_fold()

		# Drawing hands: semi-bluff if good draw; otherwise respect pot odds
		if self._has_strong_draw(all_cards):
			to_call = GameInfoAPI.calculate_bet_amount(game_state.current_bet, game_state.player_bets.get(self.name, 0))
			pot_odds = GameInfoAPI.get_pot_odds(game_state.pot, to_call)
			# If free or good odds, call; if we can apply pressure, semi-bluff
			if to_call == 0 and PlayerAction.RAISE in legal_actions:
				return PlayerAction.RAISE, self._choose_raise_amount(game_state, min_bet, max_bet, factor=0.6)
			# require ~3:1 or better for draws
			if pot_odds >= 3.0 and PlayerAction.CALL in legal_actions:
				return PlayerAction.CALL, 0
			# else fold or check
			return fallback_check_call_fold()

		# Nothing: check or fold
		return fallback_check_call_fold()

	def _choose_raise_amount(self, game_state: GameState, min_bet: int, max_bet: int, factor: float = 1.0) -> int:
		"""Choose a sensible raise amount: factor * pot, clamped to [min_bet, max_bet].
		The game expects the total bet amount (not additional)."""
		desired = int(game_state.pot * factor)
		amount = max(min_bet, desired)
		amount = min(amount, max_bet)
		if amount < min_bet:
			return min_bet
		return amount

	def _has_strong_draw(self, all_cards: List[Card]) -> bool:
		# Flush draw: 4 to a suit
		suits = [c.suit for c in all_cards]
		for s in set(suits):
			if suits.count(s) >= 4:
				return True

		# Open-ended straight draw approximation: check if there are 4 cards in sequence window
		ranks = sorted(set(c.rank.value for c in all_cards))
		if len(ranks) >= 4:
			for i in range(len(ranks) - 3):
				if ranks[i+3] - ranks[i] == 3:
					return True
		# Ace-low considerations
		if set([14, 2, 3, 4]).issubset(ranks) or set([2, 3, 4, 5]).issubset(ranks):
			return True

		return False

	def _should_bluff_or_value(self, pot: int, rank_value: int) -> bool:
		# Decide whether to raise for value or as bluff/semi-bluff.
		# Use simple heuristics: higher aggression when pot is larger or we have stronger hands.
		if rank_value >= HandEvaluator.HAND_RANKINGS.get('pair', 1):
			return True if (pot > 2 * max(1, self.aggression * 10)) else False
		# otherwise use aggression multiplier
		return True if self.aggression > 0.6 else False

	def hand_complete(self, game_state: GameState, hand_result: Dict[str, Any]):
		self.hands_played += 1
		if 'winners' in hand_result and self.name in hand_result['winners']:
			self.hands_won += 1
			# Slightly increase aggression on wins
			self.aggression = min(0.8, self.aggression + 0.01)
		else:
			# Slightly reduce aggression on losses
			self.aggression = max(0.3, self.aggression - 0.005)

		# Update simple opponent stats based on final bets seen
		try:
			for player, bet in game_state.player_bets.items():
				if player == self.name:
					continue
				stats = self.opponent_stats.setdefault(player, {'seen': 0, 'raised': 0, 'won': 0})
				stats['seen'] += 1
				if bet > game_state.big_blind * 2:
					stats['raised'] += 1
				if 'winners' in hand_result and player in hand_result['winners']:
					stats['won'] += 1
		except Exception:
			pass

	def tournament_start(self, players: List[str], starting_chips: int):
		super().tournament_start(players, starting_chips)
		# Reset counters
		self.hands_played = 0
		self.hands_won = 0
		# Adjust base aggression by table size
		if len(players) <= 4:
			self.aggression = 0.65
		elif len(players) >= 8:
			self.aggression = 0.45
		else:
			self.aggression = 0.55

