#!/usr/bin/env python3
"""
Visualize lifetime events from simulation log using Mermaid diagram.
"""
import re
import sys
from pathlib import Path


def parse_person_id(person_str):
    """
    Parse person ID string like '0000000031^(173)' or '0000000031^(-1)'.
    Returns (base_id, sample_num) where sample_num is None if -1.
    """
    match = re.match(r"(\d+)\^\((-?\d+)\)", person_str)
    if match:
        base_id = match.group(1)
        sample_num = int(match.group(2))
        return base_id, sample_num if sample_num != -1 else None
    return person_str, None


def parse_log(log_file):
    """Parse the log file and extract events."""
    events = []

    with open(log_file, "r") as f:
        for line in f:
            # Skip header and footer lines
            if "ready to go" in line or "finished" in line or "type       time" in line:
                continue

            # Parse happened event lines
            match = re.match(
                r"\s*event:happ/\s+(\w+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\S+)\s+(\d+)",
                line,
            )
            if match:
                person_str = match.group(4)
                base_id, sample_num = parse_person_id(person_str)
                events.append(
                    {
                        "status": "happened",
                        "type": match.group(1),
                        "time": float(match.group(2)),
                        "duration": float(match.group(3)),
                        "person": base_id,
                        "person_full": person_str,
                        "sample": sample_num,
                        "priority": int(match.group(5)),
                        "canceled_by": None,
                    }
                )
                continue

            # Parse generated event lines
            match = re.match(
                r"\s+\|-:gene/\s+(\w+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\S+)\s+(\d+)",
                line,
            )
            if match:
                person_str = match.group(4)
                base_id, sample_num = parse_person_id(person_str)
                events.append(
                    {
                        "status": "generated",
                        "type": match.group(1),
                        "time": float(match.group(2)),
                        "duration": float(match.group(3)),
                        "person": base_id,
                        "person_full": person_str,
                        "sample": sample_num,
                        "priority": int(match.group(5)),
                        "canceled_by": None,
                    }
                )
                continue

            # Parse canceled event lines
            match = re.match(
                r"\s*event:canc/\s+(\w+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\S+)\s+(\d+)\s+cancelled by\s+(\S+)",
                line,
            )
            if match:
                person_str = match.group(4)
                base_id, sample_num = parse_person_id(person_str)
                canceled_by_ref = match.group(6)
                events.append(
                    {
                        "status": "canceled",
                        "type": match.group(1),
                        "time": float(match.group(2)),
                        "duration": float(match.group(3)),
                        "person": base_id,
                        "person_full": person_str,
                        "sample": sample_num,
                        "priority": int(match.group(5)),
                        "canceled_by": canceled_by_ref,
                    }
                )
                continue

    return events


def get_event_color(event_type):
    """Get color for different event types."""
    colors = {
        "endpr": "#90EE90",  # Light green
        "offnd": "#FFB6C6",  # Light pink
        "retrn": "#87CEEB",  # Light blue
        "leave": "#FFA500",  # Orange
        "incar": "#DC143C",  # Crimson
        "arriv": "#9370DB",  # Medium purple - arrival
        "admit": "#32CD32",  # Lime green - admitted to program
        "revok": "#FF6347",  # Tomato red - revoked from program
    }
    return colors.get(event_type, "#E0E0E0")


def generate_mermaid(events, person_id=None):
    """Generate Mermaid diagram from events with timeline layout."""

    # Filter events by base person ID if specified
    if person_id:
        # Extract base ID from input (in case user provides full format)
        base_id, _ = parse_person_id(person_id)
        if base_id == person_id:
            # Input was just the base ID, use as-is
            base_id = person_id
        events = [e for e in events if e.get("person") == base_id]

    if not events:
        return "graph TD\n    Start([No events found])"

    # Create node map: use (time, type, status) to create unique nodes
    # But merge happened events with their corresponding generated events
    node_map = {}  # event index -> node_id
    node_info = {}  # node_id -> info dict

    # First pass: assign node IDs
    # Key by (time, type) - merge happened/generated events at same time/type
    time_type_to_node = {}

    for i, event in enumerate(events):
        key = (event["time"], event["type"])

        if key not in time_type_to_node:
            node_id = f"N{len(time_type_to_node)}"
            time_type_to_node[key] = node_id
            node_info[node_id] = {
                "time": event["time"],
                "type": event["type"],
                "status": event["status"],
                "canceled_by": event.get("canceled_by"),
                "sample": event.get("sample"),  # Track sample number for arrivals
            }
        else:
            # Update status - canceled always wins (it's the final state)
            node_id = time_type_to_node[key]
            if event["status"] == "canceled":
                node_info[node_id]["status"] = "canceled"
                node_info[node_id]["canceled_by"] = event.get("canceled_by")
            elif (
                event["status"] == "happened"
                and node_info[node_id]["status"] == "generated"
            ):
                node_info[node_id]["status"] = "happened"
            # Update sample if this event has one
            if event.get("sample") is not None:
                node_info[node_id]["sample"] = event.get("sample")

        node_map[i] = time_type_to_node[key]

    # Build mermaid diagram
    mermaid_lines = ["graph TD"]
    mermaid_lines.append("    Start([Start])")

    # Define nodes
    for node_id, info in node_info.items():
        # For arrival events, show sample number
        if info["type"] == "arriv" and info.get("sample") is not None:
            label = f"arriv<br/>T={info['time']:.0f}<br/>sample={info['sample']}"
        else:
            label = f"{info['type']}<br/>T={info['time']:.0f}"
        status = info["status"]

        if status == "happened":
            if info["type"] == "leave":
                mermaid_lines.append(f"    {node_id}([{label}])")  # Stadium
            elif info["type"] == "arriv":
                mermaid_lines.append(
                    f"    {node_id}>{label}]"
                )  # Asymmetric for arrival
            elif info["type"] in ("admit", "revok"):
                mermaid_lines.append(f"    {node_id}(({label}))")  # Circle
            else:
                mermaid_lines.append(f"    {node_id}[{label}]")  # Box
        elif status == "canceled":
            mermaid_lines.append(f"    {node_id}{{{label}}}")  # Diamond/Rhombus
        else:  # generated
            mermaid_lines.append(f"    {node_id}[/{label}/]")  # Parallelogram

    # Create main timeline (only truly happened events, not canceled ones)
    # Sort all events by time, then filter to those that are still "happened"
    timeline_nodes = []
    for node_id, info in sorted(node_info.items(), key=lambda x: x[1]["time"]):
        if info["status"] == "happened":
            if node_id not in timeline_nodes:
                timeline_nodes.append(node_id)

    prev_node = "Start"
    for node_id in timeline_nodes:
        mermaid_lines.append(f"    {prev_node} ==> {node_id}")
        prev_node = node_id

    # Group nodes by time for vertical alignment
    nodes_by_time = {}
    for node_id, info in node_info.items():
        time_int = int(info["time"])
        if time_int not in nodes_by_time:
            nodes_by_time[time_int] = []
        nodes_by_time[time_int].append(node_id)

    # Add Start to time 0
    if 0 not in nodes_by_time:
        nodes_by_time[0] = []
    nodes_by_time[0].append("Start")

    # Create subgraphs for time alignment
    for time in sorted(nodes_by_time.keys()):
        nodes = list(set(nodes_by_time[time]))
        if nodes:
            mermaid_lines.append(f"    subgraph T{time}[ ]")
            mermaid_lines.append(f"        direction LR")
            for node in nodes:
                mermaid_lines.append(f"        {node}")
            mermaid_lines.append("    end")

    # Note: Removed invisible links (~~~) as they may cause syntax errors in some Mermaid versions
    # The subgraphs and main timeline arrows should provide sufficient ordering

    # Add generation arrows (happened event -> generated events that follow it)
    for i, event in enumerate(events):
        if event["status"] != "happened":
            continue

        # Find generated events immediately following this happened event
        j = i + 1
        while j < len(events) and events[j]["status"] == "generated":
            src_node = node_map[i]
            dst_node = node_map[j]
            if src_node != dst_node:
                mermaid_lines.append(f"    {src_node} -.-> {dst_node}")
            j += 1

    # Add cancellation arrows from the explicit "cancelled by" info
    for node_id, info in node_info.items():
        if info["status"] == "canceled" and info.get("canceled_by"):
            # Parse canceled_by: format is "time_personid_type" e.g., "716.80_211_retrn"
            ref = info["canceled_by"]
            parts = ref.rsplit(
                "_", 2
            )  # Split from right to get time, person_suffix, type
            if len(parts) >= 2:
                try:
                    cancel_time = float(parts[0])
                    cancel_type = parts[-1]

                    # Find the canceling node
                    cancel_key = (cancel_time, cancel_type)
                    if cancel_key in time_type_to_node:
                        cancel_node = time_type_to_node[cancel_key]
                        mermaid_lines.append(
                            f"    {cancel_node} -. cancels .-> {node_id}"
                        )
                except (ValueError, IndexError):
                    pass

    # Add styling
    mermaid_lines.append("")
    for node_id, info in node_info.items():
        color = get_event_color(info["type"])
        status = info["status"]

        if status == "happened":
            mermaid_lines.append(
                f"    style {node_id} fill:{color},stroke:#333,stroke-width:3px"
            )
        elif status == "canceled":
            mermaid_lines.append(
                f"    style {node_id} fill:#FFD0D0,stroke:#FF0000,stroke-width:3px"
            )
        else:  # generated
            mermaid_lines.append(
                f"    style {node_id} fill:#F0F0F0,stroke:#999,stroke-width:2px,stroke-dasharray: 5 5"
            )

    return "\n".join(mermaid_lines)


def get_assets_dir():
    """Get the assets directory path."""
    return Path(__file__).parent / "assets"


def generate_html(mermaid_code, person_id=None, embed_assets=True):
    """Generate HTML file with embedded Mermaid diagram."""
    title = (
        f"Lifetime Visualization - Person {person_id}"
        if person_id
        else "Lifetime Visualization"
    )

    assets_dir = get_assets_dir()
    template_path = assets_dir / "visualize.html"
    css_path = assets_dir / "visualize.css"

    if template_path.exists() and css_path.exists():
        # Load template and CSS
        template = template_path.read_text()
        css = css_path.read_text()

        # Replace placeholders
        html = template.replace("{{title}}", title)
        html = html.replace("{{mermaid_code}}", mermaid_code)

        if embed_assets:
            # Embed CSS inline for standalone HTML files
            html = html.replace(
                '<link rel="stylesheet" href="assets/visualize.css">',
                f"<style>\n{css}\n    </style>",
            )

        return html
    else:
        # Fallback to inline template if assets not found
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        .mermaid-container {{ width: 100%; margin: 20px 0; border: 1px solid #ddd; border-radius: 8px; background: white; padding: 20px; overflow-x: auto; }}
        .mermaid {{ display: inline-block; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="mermaid-container">
            <div class="mermaid">
{mermaid_code}
            </div>
        </div>
    </div>
    <script>mermaid.initialize({{ startOnLoad: true, theme: 'default' }});</script>
</body>
</html>
"""


def main():
    if len(sys.argv) < 2:
        print("Usage: python log_visualize.py <log_file> [person_id] [-o output.html]")
        sys.exit(1)

    log_file = sys.argv[1]
    person_id = None
    output_file = None

    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "-o" and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        else:
            person_id = sys.argv[i]
            i += 1

    if not Path(log_file).exists():
        print(f"Error: File '{log_file}' not found")
        sys.exit(1)

    events = parse_log(log_file)

    if not events:
        print("No events found in log file")
        sys.exit(1)

    mermaid_code = generate_mermaid(events, person_id)

    # If output file specified, save as HTML
    if output_file:
        html_content = generate_html(mermaid_code, person_id)
        with open(output_file, "w") as f:
            f.write(html_content)
        print(f"✓ Saved visualization to {output_file}")
        print(f"  Open the file in your browser to view the diagram.")
    else:
        # Print to console in markdown format
        print("```mermaid")
        print(mermaid_code)
        print("```")


if __name__ == "__main__":
    main()
