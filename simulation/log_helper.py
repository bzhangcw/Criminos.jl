#!/usr/bin/env python3
"""
Helper tool to analyze simulation log files and show statistics.
"""
import re
import sys
from pathlib import Path
from collections import defaultdict


def parse_log(log_file):
    """Parse the log file and extract events."""
    events = []

    with open(log_file, "r") as f:
        for line in f:
            # Skip header and footer lines
            if "ready to go" in line or "finished" in line or "type       time" in line:
                continue

            # Parse event lines (both happened and generated)
            match = re.match(
                r"\s*(event:happ/|\s+\|-:gene/)\s+(\w+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\S+)\s+(\d+)",
                line,
            )
            if match:
                event_type = match.group(1).strip()
                event_name = match.group(2)
                time = float(match.group(3))
                duration = float(match.group(4))
                person = match.group(5)
                priority = int(match.group(6))

                is_happened = "event:happ" in event_type

                events.append(
                    {
                        "type": event_name,
                        "time": time,
                        "duration": duration,
                        "person": person,
                        "priority": priority,
                        "is_happened": is_happened,
                    }
                )

            # Parse update score lines
            match_score = re.match(r"event:happ/\s+(\d+):\s+(.+)", line)
            if match_score:
                events.append(
                    {
                        "type": "score",
                        "message": match_score.group(2),
                        "is_happened": True,
                    }
                )

    return events


def analyze_events(events):
    """Analyze events and compute statistics."""
    stats = {
        "total_events": 0,
        "happened_events": 0,
        "generated_events": 0,
        "persons": set(),
        "person_details": defaultdict(
            lambda: {
                "first_seen": None,
                "events": defaultdict(int),
                "happened_events": 0,
                "generated_events": 0,
            }
        ),
        "event_types": defaultdict(int),
    }

    for event in events:
        if event["type"] == "score":
            continue

        stats["total_events"] += 1

        if event["is_happened"]:
            stats["happened_events"] += 1
        else:
            stats["generated_events"] += 1

        person = event["person"]
        stats["persons"].add(person)

        # Track per-person statistics
        person_stats = stats["person_details"][person]
        if person_stats["first_seen"] is None:
            person_stats["first_seen"] = event["time"]

        person_stats["events"][event["type"]] += 1

        if event["is_happened"]:
            person_stats["happened_events"] += 1
        else:
            person_stats["generated_events"] += 1

        # Track event types
        stats["event_types"][event["type"]] += 1

    return stats


def print_statistics(stats, show_persons=False):
    """Print statistics in a formatted way."""
    print("\n" + "=" * 60)
    print("SIMULATION LOG STATISTICS")
    print("=" * 60)

    print(f"\n📊 Overall Statistics:")
    print(f"  Total Events:      {stats['total_events']}")
    print(f"  Happened Events:   {stats['happened_events']}")
    print(f"  Generated Events:  {stats['generated_events']}")
    print(f"  Total Persons:     {len(stats['persons'])}")

    print(f"\n📈 Event Types:")
    for event_type, count in sorted(stats["event_types"].items()):
        print(f"  {event_type:8s}: {count:4d}")

    if show_persons:
        print(f"\n👥 Persons List:")
        print(
            f"  {'Person ID':<25s} {'First Seen':>12s} {'Happened':>10s} {'Generated':>10s} {'Events'}"
        )
        print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*10} {'-'*30}")

        # Sort persons by first appearance time
        sorted_persons = sorted(
            stats["person_details"].items(), key=lambda x: x[1]["first_seen"]
        )

        for person, details in sorted_persons:
            first_seen = f"{details['first_seen']:.1f}"
            happened = details["happened_events"]
            generated = details["generated_events"]

            # Format event counts
            event_summary = ", ".join(
                [
                    f"{event_type}:{count}"
                    for event_type, count in sorted(details["events"].items())
                ]
            )

            print(
                f"  {person:<25s} {first_seen:>12s} {happened:>10d} {generated:>10d} {event_summary}"
            )

        print(f"\n  Total: {len(stats['persons'])} person(s)")

    print("\n" + "=" * 60 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python log_helper.py <log_file> [-l]")
        print("\nOptions:")
        print("  -l    Show detailed list of all persons")
        sys.exit(1)

    log_file = sys.argv[1]
    show_persons = "-l" in sys.argv

    if not Path(log_file).exists():
        print(f"Error: File '{log_file}' not found")
        sys.exit(1)

    events = parse_log(log_file)

    if not events:
        print("No events found in log file")
        sys.exit(1)

    stats = analyze_events(events)
    print_statistics(stats, show_persons)


if __name__ == "__main__":
    main()

