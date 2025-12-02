#!/usr/bin/env python3
"""
Script to run multiple tournaments and aggregate results
"""
import collections
import time
import sys
import logging
from tournament_runner import TournamentRunner, TournamentSettings, TournamentType

def run_many(num_tournaments=1000):
    print(f"Running {num_tournaments} tournaments...")
    print("=" * 40)
    
    # Disable logging to avoid spam and speed up execution
    # We configure basicConfig with a NullHandler so TournamentRunner's basicConfig call does nothing.
    logging.basicConfig(handlers=[logging.NullHandler()], level=logging.CRITICAL)
    # Also force set level for existing loggers just in case
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # Default settings (matching run_tournament.py)
    settings = TournamentSettings(
        tournament_type=TournamentType.FREEZE_OUT,
        starting_chips=1000,
        small_blind=10,
        big_blind=20,
        time_limit_per_action=10.0,
        blind_increase_interval=10,
        blind_increase_factor=1.5
    )
    
    win_counts = collections.Counter()
    disqualified_counts = collections.Counter()
    total_points = collections.Counter()
    podium_points = collections.Counter()
    total_earnings = collections.Counter()
    total_hands_played = 0
    start_time = time.time()
    
    # Configure stdout for Windows compatibility
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    def calculate_payouts(num_players, entry_fee=1000):
        pool = num_players * entry_fee
        payouts = {} # position -> amount
        
        if num_players < 8:
            # Standard top 3 split for small tourneys
            payouts[1] = int(pool * 0.50)
            payouts[2] = int(pool * 0.30)
            payouts[3] = int(pool * 0.20)
        else:
            # 8th gets entry back
            # 1-7 get entry back + share of excess
            base_payout = 1000
            excess = pool - (8 * base_payout)
            
            # Skewed weights for 1-7
            weights = [45, 25, 15, 8, 4, 2, 1] # Sum = 100
            
            for i in range(7):
                pos = i + 1
                bonus = int(excess * (weights[i] / 100.0))
                payouts[pos] = base_payout + bonus
            
            payouts[8] = base_payout
            
        return payouts

    try:
        for i in range(num_tournaments):
            # Show progress every 10 runs
            if (i + 1) % 10 == 0:
                print(f"Running tournament {i + 1}/{num_tournaments}...", end='\r')
                
            # Create a new runner for each tournament to ensure clean state
            runner = TournamentRunner(settings, "players", "logs")
            
            results = runner.run_tournament()
            
            # Track winner
            if results['final_standings']:
                winner_name = results['final_standings'][0][0]
                win_counts[winner_name] += 1
                
                standings = results['final_standings']
                num_players = len(standings)
                
                # Calculate payouts for this tournament
                payout_structure = calculate_payouts(num_players)
                
                for player_name, chips, position in standings:
                    # Deduct entry fee
                    total_earnings[player_name] -= 1000
                    
                    # Add prize money if any
                    if position in payout_structure:
                        total_earnings[player_name] += payout_structure[position]
                    
                    # Standard Points
                    points = num_players - position
                    total_points[player_name] += points
                    
                    # Podium Points
                    if position == 1:
                        podium_points[player_name] += 3
                    elif position == 2:
                        podium_points[player_name] += 2
                    elif position == 3:
                        podium_points[player_name] += 1
            
            # Track disqualifications
            for bot_name, stats in results['bot_stats'].items():
                if stats['is_disqualified']:
                    disqualified_counts[bot_name] += 1
            
            total_hands_played += results['total_hands']
            
    except KeyboardInterrupt:
        print("\nStopping early...")
    except Exception as e:
        print(f"\nError during execution: {e}")
        
    end_time = time.time()
    duration = end_time - start_time
    
    num_runs = sum(win_counts.values())
    
    print("\n" + "="*115)
    print(f"AGGREGATE RESULTS ({num_runs} tournaments run)")
    print("="*115)
    print(f"Total Duration: {duration:.2f}s ({duration/num_runs:.2f}s per run)")
    print(f"Avg Hands per Tournament: {total_hands_played / num_runs:.1f}")
    print("-" * 115)
    print(f"{ 'Bot Name':<30} | {'Earnings':<12} | {'Points':<8} | {'Avg Pts':<8} | {'Podium':<8} | {'Wins':<8} | {'Win %':<8} | {'DQ':<5}")
    print("-" * 115)
    
    # Sort by Earnings
    sorted_bots = total_earnings.most_common()
    
    # Include bots that have 0 points (if any, e.g. always last place or DQ'd before play)
    all_bots = set(disqualified_counts.keys()) | set(total_points.keys())
    for bot in all_bots:
        if bot not in total_earnings:
            sorted_bots.append((bot, 0))
            
    # Re-sort to be sure if we appended
    sorted_bots.sort(key=lambda x: x[1], reverse=True)
    
    for bot_name, earnings in sorted_bots:
        wins = win_counts[bot_name]
        win_rate = (wins / num_runs) * 100 if num_runs > 0 else 0
        points = total_points[bot_name]
        avg_points = points / num_runs if num_runs > 0 else 0
        p_points = podium_points[bot_name]
        dqs = disqualified_counts[bot_name]
        
        earnings_str = f"{earnings:,}"
        
        print(f"{bot_name:<30} | {earnings_str:<12} | {points:<8} | {avg_points:<8.2f} | {p_points:<8} | {wins:<8} | {win_rate:>7.1f}% | {dqs:<5}")

if __name__ == "__main__":
    run_many(1000)
