import pandas as pd # type: ignore
import random
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

# Define safer intensity ranges (avoid training too close to failure)
INTENSITY_ZONES = {
    "very_low": {"percentage": "50-55% 1RM", "reps_per_set": "6-8"},
    "low": {"percentage": "~60% 1RM", "reps_per_set": "5-6"},
    "medium": {"percentage": "70-75% 1RM", "reps_per_set": "4-5"},
    "high": {"percentage": "75-85% 1RM", "reps_per_set": "3-4"}  # Capped at 85% for safety
}

# Main volume distribution variants across 4 weeks
# This dictionary defines intensity zones based on percentage of one-rep max (1RM) and recommended repetitions per set.
MAIN_VARIANT = {
    1: [0.35, 0.28, 0.22, 0.15], 
    2: [0.15, 0.35, 0.28, 0.22], 
    3: [0.28, 0.35, 0.22, 0.15], 
    4: [0.22, 0.35, 0.28, 0.15], 
    5: [0.15, 0.22, 0.35, 0.28], 
    6: [0.22, 0.28, 0.35, 0.15], 
    7: [0.15, 0.28, 0.35, 0.22], 
    8: [0.35, 0.15, 0.28, 0.22], 
    9: [0.35, 0.22, 0.28, 0.15], 
    10: [0.28, 0.15, 0.35, 0.22], 
    11: [0.28, 0.22, 0.35, 0.15], 
    12: [0.15, 0.22, 0.28, 0.35], 
    13: [0.15, 0.35, 0.22, 0.28], 
    14: [0.22, 0.35, 0.15, 0.28], 
    15: [0.22, 0.28, 0.15, 0.35], 
    16: [0.15, 0.28, 0.22, 0.35],     
}

# Separate intensity variants (using same patterns but possibly different selections)
INTENSITY_VARIANT = MAIN_VARIANT.copy()
# Add some extra intensity variants
additional_variants = {
    17: [0.35, 0.28, 0.15, 0.22],
    18: [0.35, 0.15, 0.22, 0.28],
    19: [0.35, 0.22, 0.15, 0.28],
    20: [0.28, 0.35, 0.15, 0.22],
    21: [0.22, 0.15, 0.35, 0.28],
    22: [0.28, 0.15, 0.22, 0.35],
    23: [0.28, 0.22, 0.15, 0.35],
    24: [0.22, 0.15, 0.28, 0.35],
}
INTENSITY_VARIANT.update(additional_variants)

# Session distribution patterns
WEEK_DISTRIBUTION = [
    [0.4, 0.6],  # 2 sessions per week
    [0.25, 0.33, 0.42],  # 3 sessions per week
    [0.15, 0.22, 0.28, 0.35]  # 4 sessions per week
]

# Exercise templates with recommended parameters
# Patterns for distributing volume across different sessions within a week.
EXERCISE_TEMPLATES = {
    "squat": {
        "volume_range": (150, 250),
        "intensity_distribution": {"very_low": 0.0, "low": 0.0, "medium": 0.7, "high": 0.3},
        "sessions_per_week": 3,
        "recovery_needs": "high"
    },
    "powerclean": {
        "volume_range": (150, 250),
        "intensity_distribution": {"very_low": 0.0, "low": 0.0, "medium": 0.8, "high": 0.2},
        "sessions_per_week": 3,
        "recovery_needs": "high"
    },
    "bench_press": {
        "volume_range": (200, 400),
        "intensity_distribution": {"very_low": 0.0, "low": 0.0, "medium": 0.7, "high": 0.3},
        "sessions_per_week": 3,
        "recovery_needs": "medium"
    },
    "overhead_press": {
        "volume_range": (180, 300),
        "intensity_distribution": {"very_low": 0.0, "low": 0.0, "medium": 0.7, "high": 0.3},
        "sessions_per_week": 3,
        "recovery_needs": "medium-high"
    },
    "pull_up": {
        "volume_range": (200, 400),
        "intensity_distribution": {"very_low": 0.0, "low": 0.0, "medium": 0.7, "high": 0.3},
        "sessions_per_week": 3,
        "recovery_needs": "medium"
    },
    "dip": {
        "volume_range": (200, 400),
        "intensity_distribution": {"very_low": 0.0, "low": 0.0, "medium": 0.7, "high": 0.3},
        "sessions_per_week": 3, 
        "recovery_needs": "medium"
    }
}
    
def make_schedule(exercise_name=None, custom_distribution=None, total_volume=None, sessions_per_week=3, 
                 visualize=True, export_to_excel=False, excel_writer=None):
    """
    Generate a 4-week Soviet-style training schedule.
    
    Parameters:
    -----------
    exercise_name : str
        Name of exercise from templates or custom
    custom_distribution : dict
        Custom intensity distribution if not using a template
        Format: {"very_low": x, "low": y, "medium": z, "high": w} where x+y+z+w=1
    total_volume : int
        Monthly total volume (if None, uses template recommendation)
    sessions_per_week : int
        Number of training sessions per week (1-4)
    visualize : bool
        Whether to display visualizations of the schedule
    export_to_excel : bool
        Whether to export the schedule to Excel
    strict_custom_distribution : bool
        If True, strictly adheres to the custom intensity distribution without adjustments
        
    Returns:
    --------
    DataFrame with the full training schedule
    """
    # Input validation
    if exercise_name is not None and exercise_name not in EXERCISE_TEMPLATES and custom_distribution is None:
        print(f"Exercise '{exercise_name}' not found in templates. Please provide custom_distribution.")
        print(f"Available templates: {list(EXERCISE_TEMPLATES.keys())}")
        return None
    
    # Set distribution based on exercise template or custom values
    if exercise_name is not None and exercise_name in EXERCISE_TEMPLATES:
        template = EXERCISE_TEMPLATES[exercise_name]
        intensity_dist = custom_distribution if custom_distribution is not None else template["intensity_distribution"]
        if total_volume is None:
            # Select a random volume within the recommended range
            vol_min, vol_max = template["volume_range"]
            total_volume = random.randint(vol_min, vol_max)
        if sessions_per_week is None:
            sessions_per_week = template["sessions_per_week"]
    elif custom_distribution is not None:
        intensity_dist = custom_distribution
        if total_volume is None:
            # default to a random volume between 200 and 300 reps
            total_volume = random.randint(200, 300)
    else:
        print("Error: Must provide either a valid exercise_name or custom_distribution")
        return None

    # Check that intensity distribution sums to 1
    if abs(sum(intensity_dist.values()) - 1.0) > 0.01:
        print(f"Warning: Intensity distribution sums to {sum(intensity_dist.values())}, not 1.0. Normalizing...")
        total = sum(intensity_dist.values())
        for key in intensity_dist:
            intensity_dist[key] /= total
    
    # Select random variation patterns for volume and intensity
    main_variant_key = random.choice(list(MAIN_VARIANT.keys()))
    main_variants = MAIN_VARIANT[main_variant_key]
    
    # We'll still use intensity variants for display, but won't override custom distribution
    intensity_variant_key = random.choice(list(INTENSITY_VARIANT.keys()))
    intensity_variants = INTENSITY_VARIANT[intensity_variant_key]
    
    # Select session distribution based on sessions per week
    if sessions_per_week < 1 or sessions_per_week > 4:
        print("Warning: Sessions per week must be 1-4, defaulting to 3")
        sessions_per_week = 3
    
    # Determine session distribution pattern
    if sessions_per_week > 1:
        session_distribution = WEEK_DISTRIBUTION[sessions_per_week - 2]
    else:
        session_distribution = [1.0]
    
    # Create schedule dataframe
    schedule_data = []
    
    # Extract intensity zone percentages
    very_low_pct = intensity_dist.get("very_low", 0)
    low_pct = intensity_dist.get("low", 0)
    medium_pct = intensity_dist.get("medium", 0)
    high_pct = intensity_dist.get("high", 0)
    
    strict_custom_distribution = (very_low_pct + low_pct + medium_pct + high_pct) > 0
    # Generate weekly schedules
    for week in range(4):
        # Weekly volume calculation
        week_volume = round(total_volume * main_variants[week])
        
        # Calculate raw intensity distribution for this week
        very_low_volume = round(very_low_pct * week_volume)
        low_volume = round(low_pct * week_volume)
        medium_volume = round(medium_pct * week_volume)
        high_volume = round(high_pct * week_volume)
        
        # Adjust for rounding errors
        total_adjusted = very_low_volume + low_volume + medium_volume + high_volume
        if total_adjusted != week_volume:
            diff = week_volume - total_adjusted
            # Add or subtract the difference from the largest non-zero intensity
            if high_pct > 0:
                high_volume += diff
            elif medium_pct > 0:
                medium_volume += diff
            elif low_pct > 0:
                low_volume += diff
            else:
                very_low_volume += diff
        
        # Create session distribution based on weekly volumes
        session_volumes = []
        unallocated_volume = {"very_low": very_low_volume, "low": low_volume, 
                           "medium": medium_volume, "high": high_volume}
        
        for session in range(sessions_per_week):
            session_volume = round(week_volume * session_distribution[session])
            
            # Initialize session intensities
            session_intensities = {"very_low": 0, "low": 0, "medium": 0, "high": 0}
            
            if strict_custom_distribution:
                # Calculate each intensity's volume for this session proportionally
                for intensity in ["very_low", "low", "medium", "high"]:
                    if intensity_dist.get(intensity, 0) > 0:
                        session_intensities[intensity] = round(session_volume * intensity_dist.get(intensity, 0))
                
                # Adjust for rounding errors
                total_session = sum(session_intensities.values())
                if total_session != session_volume:
                    diff = session_volume - total_session
                    # Add or subtract from largest intensity to maintain total
                    highest_intensity = max(session_intensities, key=session_intensities.get)
                    session_intensities[highest_intensity] += diff
            else:
                # Original approach - distribute intensities based on session number
                # This gives higher intensity to earlier sessions
                remaining_volume = session_volume
                
                # Prioritize intensities based on session number
                intensity_order = []
                if session == 0:
                    intensity_order = ["high", "medium", "low", "very_low"]
                elif session == 1:
                    intensity_order = ["medium", "high", "low", "very_low"]
                else:
                    intensity_order = ["low", "medium", "high", "very_low"]
                
                # Assign volumes in priority order
                for intensity in intensity_order:
                    available = unallocated_volume[intensity]
                    to_assign = min(available, remaining_volume)
                    session_intensities[intensity] = to_assign
                    unallocated_volume[intensity] -= to_assign
                    remaining_volume -= to_assign
                    if remaining_volume <= 0:
                        break
            
            # Add to schedule
            schedule_data.append({
                "week": week + 1,
                "session": session + 1,
                "total_reps": session_volume,
                "high_reps": session_intensities["high"],
                "medium_reps": session_intensities["medium"],
                "low_reps": session_intensities["low"],
                "very_low_reps": session_intensities["very_low"]
            })

    # Check if total distributions match expected
    total_high = sum(item["high_reps"] for item in schedule_data)
    total_medium = sum(item["medium_reps"] for item in schedule_data)
    total_low = sum(item["low_reps"] for item in schedule_data)
    total_very_low = sum(item["very_low_reps"] for item in schedule_data)
    
    expected_high = round(total_volume * high_pct)
    expected_medium = round(total_volume * medium_pct)
    expected_low = round(total_volume * low_pct)
    expected_very_low = round(total_volume * very_low_pct)
    
    print(f"Expected vs Actual Distribution:")
    print(f"HIGH: {expected_high} vs {total_high}")
    print(f"MEDIUM: {expected_medium} vs {total_medium}")
    print(f"LOW: {expected_low} vs {total_low}")
    print(f"VERY LOW: {expected_very_low} vs {total_very_low}")
    
    # Create DataFrame
    schedule_df = pd.DataFrame(schedule_data)
    
    # Print summary
    print(f"--- {exercise_name if exercise_name else 'Custom'} Training Schedule ---")
    print(f"Total monthly volume: {total_volume} reps")
    print(f"Volume distribution pattern: {main_variants}")
    print(f"Intensity distribution: {intensity_dist}")
    print(f"Sessions per week: {sessions_per_week}")
    print("\nIntensity zones:")
    for zone, details in INTENSITY_ZONES.items():
        print(f"  {zone.upper()}: {details['percentage']} for {details['reps_per_set']} reps per set")
    
    # Generate more detailed weekly breakdown
    print("\nWeekly Schedule:")
    weekly_summary = schedule_df.groupby('week').agg({
        'total_reps': 'sum',
        'high_reps': 'sum',
        'medium_reps': 'sum',
        'low_reps': 'sum',
        'very_low_reps': 'sum'
    }).reset_index()
    
    print("\nDetailed Weekly Summary:")
    for _, row in weekly_summary.iterrows():
        print(f"\nWeek {row['week']}:")
        print(f"  Total volume: {row['total_reps']} reps")
        print(f"  HIGH ({INTENSITY_ZONES['high']['percentage']}): {row['high_reps']} reps")
        print(f"  MEDIUM ({INTENSITY_ZONES['medium']['percentage']}): {row['medium_reps']} reps")
        print(f"  LOW ({INTENSITY_ZONES['low']['percentage']}): {row['low_reps']} reps")
        print(f"  VERY LOW ({INTENSITY_ZONES['very_low']['percentage']}): {row['very_low_reps']} reps")

    print("\nSession Breakdown:")
    for _, row in schedule_df.iterrows():
        print(f"Week {row['week']}, Session {row['session']}:")
        print(f"  Total: {row['total_reps']} reps")
        
    # Visualize the schedule
    if visualize:
        # Plot weekly volume
        plt.figure(figsize=(12, 10))
        
        # Weekly volume plot
        plt.subplot(2, 2, 1)
        weekly_volumes = weekly_summary['total_reps'].values
        plt.bar(range(1, 5), weekly_volumes, color='darkblue')
        plt.title('Weekly Training Volume')
        plt.xlabel('Week')
        plt.ylabel('Total Reps')
        plt.xticks(range(1, 5))
        
        # Weekly intensity distribution
        plt.subplot(2, 2, 2)
        intensity_data = weekly_summary[['high_reps', 'medium_reps', 'low_reps', 'very_low_reps']].values
        bottom = np.zeros(4)
        
        for i, intensity in enumerate(['high_reps', 'medium_reps', 'low_reps', 'very_low_reps']):
            plt.bar(range(1, 5), weekly_summary[intensity], bottom=bottom, 
                   label=intensity.replace('_reps', '').upper())
            bottom += weekly_summary[intensity].values
        
        plt.title('Weekly Intensity Distribution')
        plt.xlabel('Week')
        plt.ylabel('Reps')
        plt.xticks(range(1, 5))
        plt.legend()
        
        # Session distribution
        plt.subplot(2, 2, 3)
        for week in range(1, 5):
            week_data = schedule_df[schedule_df['week'] == week]
            x = [f"W{week}S{s}" for s in week_data['session']]
            plt.bar(x, week_data['total_reps'], color=f'C{week-1}')
        
        plt.title('Session Volume Distribution')
        plt.xlabel('Week.Session')
        plt.ylabel('Reps')
        plt.xticks(rotation=45)
        
        # Session intensity breakdown
        plt.subplot(2, 2, 4)
        
        weeks = []
        sessions = []
        for _, row in schedule_df.iterrows():
            weeks.append(row['week'])
            sessions.append(row['session'])
        
        x = [f"W{w}S{s}" for w, s in zip(weeks, sessions)]
        bottom = np.zeros(len(x))
        
        for intensity in ['high_reps', 'medium_reps', 'low_reps', 'very_low_reps']:
            plt.bar(x, schedule_df[intensity], bottom=bottom, 
                   label=intensity.replace('_reps', '').upper())
            bottom += schedule_df[intensity].values
        
        plt.title('Session Intensity Breakdown')
        plt.xlabel('Week.Session')
        plt.ylabel('Reps')
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
# Export to Excel if requested
    if export_to_excel:
        # Create workout examples with structure details
        workout_examples = []
        for _, row in schedule_df.iterrows():
            workout = {
                'week': row['week'],
                'session': row['session'],
                'total_reps': row['total_reps'],
                'workout_structure': ''
            }
            
            structure = []
            if row['high_reps'] > 0:
                high_sets = round(row['high_reps'] / 3)
                structure.append(f"{high_sets}x3 @ {INTENSITY_ZONES['high']['percentage']}")
            
            if row['medium_reps'] > 0:
                medium_sets = round(row['medium_reps'] / 4)
                structure.append(f"{medium_sets}x4 @ {INTENSITY_ZONES['medium']['percentage']}")
            
            if row['low_reps'] > 0:
                low_sets = round(row['low_reps'] / 6)
                structure.append(f"{low_sets}x6 @ {INTENSITY_ZONES['low']['percentage']}")
                
            if row['very_low_reps'] > 0:
                very_low_sets = round(row['very_low_reps'] / 8)
                structure.append(f"{very_low_sets}x8 @ {INTENSITY_ZONES['very_low']['percentage']}")
            
            workout['workout_structure'] = ', '.join(structure)
            workout_examples.append(workout)
        
        workout_examples_df = pd.DataFrame(workout_examples)
        
        # Use provided excel_writer if given, otherwise create a new one
        if excel_writer is not None:
            # Create sheet names based on exercise name
            base_name = f"{exercise_name if exercise_name else 'custom'}"
            # Make sure sheet names don't exceed Excel's 31 character limit
            if len(base_name) > 20:
                base_name = base_name[:20]
                
            # Write to the provided excel writer
            schedule_df.to_excel(excel_writer, sheet_name=f'{base_name}_Details', index=False)
            weekly_summary.to_excel(excel_writer, sheet_name=f'{base_name}_Summary', index=False)
            workout_examples_df.to_excel(excel_writer, sheet_name=f'{base_name}_Examples', index=False)
            print(f"Added {base_name} sheets to Excel workbook")
        else:
            # Create a new excel file
            filename = f"{exercise_name if exercise_name else 'custom'}_schedule.xlsx"
            with pd.ExcelWriter(filename) as writer:
                schedule_df.to_excel(writer, sheet_name='Session Details', index=False)
                weekly_summary.to_excel(writer, sheet_name='Weekly Summary', index=False)
                workout_examples_df.to_excel(writer, sheet_name='Workout Examples', index=False)
            print(f"\nExported schedule to {filename}")
    
    return schedule_df

# CALL FUNCTION
if __name__ == "__main__":
    # Example of combining multiple exercises in a single Excel file
    with pd.ExcelWriter("all_exercises_schedule.xlsx", engine="openpyxl") as writer:
        # Generate Squat schedule
        squat_schedule = make_schedule(
            "squat", 
            sessions_per_week=3, 
            visualize=True, 
            export_to_excel=True,
            excel_writer=writer
        )
        
        # Generate Bench Press schedule
        bench_schedule = make_schedule(
            "bench_press", 
            sessions_per_week=3, 
            visualize=True, 
            export_to_excel=True,
            excel_writer=writer
        )
        
        # Generate Pull Up schedule with custom distribution
        pullup_schedule = make_schedule(
            "pull_up", 
            custom_distribution={"very_low": 0, "low": 0.0, "medium": 0.7, "high": 0.3},
            sessions_per_week=3, 
            visualize=True, 
            export_to_excel=True,
            excel_writer=writer
        )
        
    print("\nExported all exercise schedules to all_exercises_schedule.xlsx")
    
    # For a single exercise example (still works as before)
    # bench_schedule = make_schedule(
    #     "bench_press", 
    #     custom_distribution={"very_low": 0, "low": 0, "medium": 0, "high": 1}, 
    #     sessions_per_week=3, 
    #     visualize=True,
    #     export_to_excel=True
    # )


#   FEATURES TO ADD
    # 1) consolidate 3 sheets per compound lift into one, and separate with empty rows. 
    # So squat, bench and pull ups would be 3 sheets total instead of the current 9.

    # 2) add modularity on sessions per week (high volume -> more sessions)

    # 3) Create a webpage for jupyter notebook, create an interactive dash with 1RM calculators

    # 4) Title for visuals, add visuals to xls file 

    # 5) telegram/discord/slack functionality