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

        person_stats = stats["person_details"][person]
        if person_stats["first_seen"] is None:
            person_stats["first_seen"] = event["time"]

        person_stats["events"][event["type"]] += 1

        if event["is_happened"]:
            person_stats["happened_events"] += 1
        else:
            person_stats["generated_events"] += 1

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

    def serve_index(self):
        """Serve the main HTML page."""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simulation Log Service</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .header h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2em;
        }
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .card h2 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.3em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        .stat-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        .stat-item:last-child {
            border-bottom: none;
        }
        .stat-label {
            color: #666;
            font-weight: 500;
        }
        .stat-value {
            color: #667eea;
            font-weight: bold;
            font-size: 1.1em;
        }
        .persons-section {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .persons-section h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.3em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        .person-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .person-card:hover {
            background: #e9ecef;
            transform: translateX(5px);
        }
        .person-id {
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
            font-family: monospace;
        }
        .person-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            font-size: 0.9em;
            color: #666;
        }
        .viz-section {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: none;
        }
        .viz-section.active {
            display: block;
        }
        .viz-section h2 {
            color: #333;
            margin-bottom: 20px;
        }
        .back-button {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            margin-bottom: 20px;
        }
        .back-button:hover {
            background: #5568d3;
        }
        .mermaid {
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        .refresh-button {
            background: #28a745;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            margin-left: 10px;
        }
        .refresh-button:hover {
            background: #218838;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Simulation Log Service</h1>
            <p>Real-time monitoring and visualization of simulation events</p>
        </div>

        <div id="main-view">
            <div class="grid" id="stats-grid">
                <div class="card">
                    <h2>📈 Overall Statistics</h2>
                    <div id="overall-stats" class="loading">Loading...</div>
                </div>
                <div class="card">
                    <h2>📊 Event Types</h2>
                    <div id="event-types" class="loading">Loading...</div>
                </div>
            </div>

            <div class="persons-section">
                <h2>👥 Persons <button class="refresh-button" onclick="loadData()">🔄 Refresh</button></h2>
                <div id="persons-list" class="loading">Loading...</div>
            </div>
        </div>

        <div id="viz-view" class="viz-section">
            <button class="back-button" onclick="showMain()">← Back to Overview</button>
            <h2 id="viz-title">Person Visualization</h2>
            <div id="viz-content"></div>
        </div>
    </div>

    <script>
        mermaid.initialize({ startOnLoad: true, theme: 'default' });

        async function loadData() {
            try {
                const statsResponse = await fetch('/api/stats');
                const stats = await statsResponse.json();
                
                // Update overall stats
                document.getElementById('overall-stats').innerHTML = `
                    <div class="stat-item">
                        <span class="stat-label">Total Events:</span>
                        <span class="stat-value">${stats.total_events}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Happened Events:</span>
                        <span class="stat-value">${stats.happened_events}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Generated Events:</span>
                        <span class="stat-value">${stats.generated_events}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Total Persons:</span>
                        <span class="stat-value">${stats.persons.length}</span>
                    </div>
                `;

                // Update event types
                let eventTypesHtml = '';
                for (const [type, count] of Object.entries(stats.event_types).sort()) {
                    eventTypesHtml += `
                        <div class="stat-item">
                            <span class="stat-label">${type}:</span>
                            <span class="stat-value">${count}</span>
                        </div>
                    `;
                }
                document.getElementById('event-types').innerHTML = eventTypesHtml;

                // Update persons list
                const personsResponse = await fetch('/api/persons');
                const persons = await personsResponse.json();
                
                let personsHtml = '';
                for (const person of persons) {
                    const eventSummary = Object.entries(person.events)
                        .map(([type, count]) => `${type}:${count}`)
                        .join(', ');
                    
                    personsHtml += `
                        <div class="person-card" onclick="showVisualization('${person.id}')">
                            <div class="person-id">${person.id}</div>
                            <div class="person-stats">
                                <div>First seen: ${person.first_seen.toFixed(1)}</div>
                                <div>Happened: ${person.happened_events}</div>
                                <div>Generated: ${person.generated_events}</div>
                                <div style="grid-column: 1/-1">${eventSummary}</div>
                            </div>
                        </div>
                    `;
                }
                document.getElementById('persons-list').innerHTML = personsHtml;

            } catch (error) {
                console.error('Error loading data:', error);
                document.getElementById('overall-stats').innerHTML = '<p style="color:red;">Error loading data</p>';
            }
        }

        async function showVisualization(personId) {
            try {
                const response = await fetch(`/api/visualize?person=${encodeURIComponent(personId)}`);
                const data = await response.json();
                
                document.getElementById('main-view').style.display = 'none';
                document.getElementById('viz-view').classList.add('active');
                document.getElementById('viz-title').textContent = `Lifetime Visualization - ${personId}`;
                
                document.getElementById('viz-content').innerHTML = `
                    <div class="mermaid">
                        ${data.mermaid}
                    </div>
                    <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                        <h3>Legend</h3>
                        <h4>Timeline (Top to Bottom)</h4>
                        <ul style="margin-left: 20px; margin-top: 10px;">
                            <li><strong>Thick arrows (==>):</strong> Main timeline - events that happened</li>
                            <li><strong>Dashed arrows (-.->):</strong> Generated/scheduled events</li>
                            <li><strong>🔴 gen leave arrows:</strong> Shows which event generated a canceled leave</li>
                            <li><strong>❌ cancels arrows:</strong> Return events that cancel leave events</li>
                        </ul>
                        <h4 style="margin-top: 15px;">Node Shapes & Colors</h4>
                        <ul style="margin-left: 20px; margin-top: 10px;">
                            <li><strong>Rectangle [Box]:</strong> Happened events (thick border)</li>
                            <li><strong>Stadium ([Box]):</strong> Leave system events</li>
                            <li><strong>Parallelogram [/Box/]:</strong> Generated events (not yet happened)</li>
                            <li><strong>Diamond {Box}:</strong> Canceled events with ❌</li>
                        </ul>
                        <h4 style="margin-top: 15px;">Event Type Colors</h4>
                        <ul style="margin-left: 20px; margin-top: 10px;">
                            <li><strong>🟢 Green:</strong> End probation (endpr)</li>
                            <li><strong>🩷 Pink:</strong> Offense (offnd)</li>
                            <li><strong>🔵 Blue:</strong> Return (retrn)</li>
                            <li><strong>🟠 Orange:</strong> Leave system (leave)</li>
                            <li><strong>⚪ Light Gray:</strong> Generated pending events</li>
                            <li><strong>🔴 Red:</strong> Canceled events</li>
                        </ul>
                    </div>
                `;
                
                mermaid.init(undefined, document.querySelectorAll('.mermaid'));
            } catch (error) {
                console.error('Error loading visualization:', error);
            }
        }

        function showMain() {
            document.getElementById('viz-view').classList.remove('active');
            document.getElementById('main-view').style.display = 'block';
        }

        // Load data on page load
        loadData();
        
        // Auto-refresh every 30 seconds
        setInterval(loadData, 30000);
    </script>
</body>
</html>"""

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

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
