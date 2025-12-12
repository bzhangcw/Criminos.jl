#!/usr/bin/env python3
"""
Web service to view simulation log statistics and visualizations.
"""
import os
import sys
import json
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import re
from collections import defaultdict

# Import visualization functions from log_visualize
from log_visualize import parse_log, generate_mermaid as generate_mermaid_viz


def analyze_events(events):
    """Analyze events and compute statistics."""
    stats = {
        "total_events": 0,
        "happened_events": 0,
        "generated_events": 0,
        "canceled_events": 0,
        "persons": set(),
        "person_details": defaultdict(
            lambda: {
                "first_seen": None,
                "events": defaultdict(int),
                "happened_events": 0,
                "generated_events": 0,
                "canceled_events": 0,
            }
        ),
        "event_types": defaultdict(int),
    }

    for event in events:
        stats["total_events"] += 1

        status = event.get("status", "happened")
        if status == "happened":
            stats["happened_events"] += 1
        elif status == "generated":
            stats["generated_events"] += 1
        elif status == "canceled":
            stats["canceled_events"] += 1

        person = event["person"]
        stats["persons"].add(person)

        person_stats = stats["person_details"][person]
        if person_stats["first_seen"] is None:
            person_stats["first_seen"] = event["time"]

        person_stats["events"][event["type"]] += 1

        if status == "happened":
            person_stats["happened_events"] += 1
        elif status == "generated":
            person_stats["generated_events"] += 1
        elif status == "canceled":
            person_stats["canceled_events"] += 1

        stats["event_types"][event["type"]] += 1

    # Convert sets to lists for JSON serialization
    stats["persons"] = sorted(list(stats["persons"]))
    stats["event_types"] = dict(stats["event_types"])

    # Convert defaultdict to regular dict for JSON
    person_details_json = {}
    for person, details in stats["person_details"].items():
        person_details_json[person] = {
            "first_seen": details["first_seen"],
            "events": dict(details["events"]),
            "happened_events": details["happened_events"],
            "generated_events": details["generated_events"],
            "canceled_events": details["canceled_events"],
        }
    stats["person_details"] = person_details_json

    return stats


class LogServiceHandler(BaseHTTPRequestHandler):
    log_file = None

    def log_message(self, format, *args):
        """Override to customize logging."""
        print(f"[{self.address_string()}] {format % args}")

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query = parse_qs(parsed_path.query)

        try:
            if path == "/" or path == "/index.html":
                self.serve_index()
            elif path.startswith("/assets/"):
                self.serve_asset(path)
            elif path == "/api/stats":
                self.serve_stats()
            elif path == "/api/persons":
                self.serve_persons()
            elif path == "/api/visualize":
                person_id = query.get("person", [None])[0]
                if person_id:
                    self.serve_visualization(person_id)
                else:
                    self.send_error(400, "Missing person parameter")
            else:
                self.send_error(404, "Not found")
        except Exception as e:
            self.send_error(500, str(e))

    def get_assets_dir(self):
        """Get the assets directory path."""
        return Path(__file__).parent / "assets"

    def serve_asset(self, path):
        """Serve static assets from assets folder."""
        # Remove /assets/ prefix
        asset_name = path[8:]  # len("/assets/") = 8
        asset_path = self.get_assets_dir() / asset_name

        if not asset_path.exists() or not asset_path.is_file():
            self.send_error(404, "Asset not found")
            return

        # Determine content type
        content_types = {
            ".css": "text/css",
            ".js": "application/javascript",
            ".html": "text/html",
        }
        ext = asset_path.suffix
        content_type = content_types.get(ext, "text/plain")

        self.send_response(200)
        self.send_header("Content-type", content_type)
        self.end_headers()
        self.wfile.write(asset_path.read_bytes())

    def serve_index(self):
        """Serve the main HTML page."""
        assets_dir = self.get_assets_dir()
        html_path = assets_dir / "service.html"

        if html_path.exists():
            html = html_path.read_text()
        else:
            # Fallback inline HTML if template not found
            html = self.get_fallback_html()

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def get_fallback_html(self):
        """Fallback HTML if templates not found."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Simulation Log Service</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
</head>
<body>
    <h1>Simulation Log Service</h1>
    <p>Assets not found. Please ensure assets folder exists.</p>
</body>
</html>"""

    def serve_stats(self):
        """Serve statistics as JSON."""
        events = parse_log(self.log_file)
        stats = analyze_events(events)

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(stats).encode())

    def serve_persons(self):
        """Serve persons list as JSON."""
        events = parse_log(self.log_file)
        stats = analyze_events(events)

        persons_list = []
        for person_id in sorted(stats["persons"]):
            details = stats["person_details"][person_id]
            persons_list.append(
                {
                    "id": person_id,
                    "first_seen": details["first_seen"],
                    "events": details["events"],
                    "happened_events": details["happened_events"],
                    "generated_events": details["generated_events"],
                }
            )

        # Sort by first appearance
        persons_list.sort(key=lambda x: x["first_seen"])

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(persons_list).encode())

    def serve_visualization(self, person_id):
        """Serve Mermaid visualization for a specific person."""
        events = parse_log(self.log_file)
        mermaid_code = generate_mermaid_viz(events, person_id)

        result = {"person": person_id, "mermaid": mermaid_code}

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())


def main():
    if len(sys.argv) < 2:
        print("Usage: python log_service.py <log_file> [port]")
        print("\nExample:")
        print("  python log_service.py results/log.dyn.log 8080")
        sys.exit(1)

    log_file = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    if not Path(log_file).exists():
        print(f"Error: File '{log_file}' not found")
        sys.exit(1)

    # Set the log file as a class variable
    LogServiceHandler.log_file = log_file

    server_address = ("", port)
    httpd = HTTPServer(server_address, LogServiceHandler)

    print("=" * 60)
    print("🚀 Simulation Log Service Started")
    print("=" * 60)
    print(f"📁 Log file: {log_file}")
    print(f"🌐 Server:   http://localhost:{port}")
    print(f"🔄 Auto-refresh: Every 30 seconds")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
        httpd.server_close()


if __name__ == "__main__":
    main()
