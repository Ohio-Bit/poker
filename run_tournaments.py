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
    
    # Disable logging ONLY if running multiple tournaments to avoid spam
    if num_tournaments > 1:
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
        
        # Top-Heavy Payout Structures (Favoring 1st/2nd, Max Top 8)
        if num_players <= 5:
            # Winner takes all
            percentages = [1.00]
        elif num_players <= 6:
            # Top 2 (70/30)
            percentages = [0.70, 0.30]
        elif num_players <= 10:
            # Top 3 (55/30/15)
            percentages = [0.55, 0.30, 0.15]
        elif num_players <= 15:
            # Top 4 (50/25/15/10)
            percentages = [0.50, 0.25, 0.15, 0.10]
        elif num_players <= 25:
            # Top 5 (45/25/15/10/5)
            percentages = [0.45, 0.25, 0.15, 0.10, 0.05]
        elif num_players <= 30:
            # Top 6 (42/24/14/10/6/4)
            percentages = [0.42, 0.24, 0.14, 0.10, 0.06, 0.04]
        elif num_players <= 49:
            # Top 7 (40/22/14/10/7/4/3)
            percentages = [0.40, 0.22, 0.14, 0.10, 0.07, 0.04, 0.03]
        else:
            # Top 8 (50+) (38/20/14/10/7/5/4/2)
            percentages = [0.38, 0.20, 0.14, 0.10, 0.07, 0.05, 0.04, 0.02]
            
        for i, pct in enumerate(percentages):
            payouts[i + 1] = int(pool * pct)
            
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
        
    try:
        end_time = time.time()
        duration = end_time - start_time
        
        num_runs = sum(win_counts.values())
        
        if num_runs == 0:
            print("No tournaments completed.")
            return

        print("\n" + "="*115)
        print(f"AGGREGATE RESULTS ({num_runs} tournaments run)")
        print("="*115)
        print(f"Total Duration: {duration:.2f}s ({duration/num_runs:.2f}s per run)")
        if num_runs > 0:
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
    except KeyboardInterrupt:
        print("\nOutput interrupted.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run multiple poker tournaments.')
    parser.add_argument('-n', '--count', type=int, default=1000, help='Number of tournaments to run')
    args = parser.parse_args()
    
    try:
        run_many(args.count)
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
        sys.exit(0)
