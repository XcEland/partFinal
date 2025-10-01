import requests
import networkx as nx
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import ttk
import json
import os
import webbrowser
import time
import http.server
import socketserver
import threading
from PIL import Image, ImageTk

# Persistent cache file
cache_file = 'distance_cache_zimbabwe.json'

# Load cache if exists, handle invalid JSON
distance_cache = {}
if os.path.exists(cache_file):
    try:
        with open(cache_file, 'r') as f:
            distance_cache = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading cache file: {e}. Resetting cache.")
        distance_cache = {}
        if os.path.exists(cache_file):
            os.remove(cache_file)

# List of 12 cities in Zimbabwe (including Beitbridge, Chiredzi, Gwanda, Hwange)
cities = [
    {"name": "Harare", "lat": -17.8277, "lon": 31.0534},
    {"name": "Bulawayo", "lat": -20.1500, "lon": 28.5833},
    {"name": "Mutare", "lat": -18.9707, "lon": 32.6695},
    {"name": "Gweru", "lat": -19.4500, "lon": 29.8167},
    {"name": "Kwekwe", "lat": -18.9281, "lon": 29.8149},
    {"name": "Kadoma", "lat": -18.3333, "lon": 29.9153},
    {"name": "Masvingo", "lat": -20.0637, "lon": 30.8277},
    {"name": "Chinhoyi", "lat": -17.3667, "lon": 30.2000},
    {"name": "Beitbridge", "lat": -22.2167, "lon": 30.0000},
    {"name": "Chiredzi", "lat": -21.0500, "lon": 31.6667},
    {"name": "Gwanda", "lat": -20.9361, "lon": 29.0069},
    {"name": "Hwange", "lat": -18.3667, "lon": 26.5000},
]

# TomTom API key
TOMTOM_API_KEY = "dzBNWMwFlqcmHWh6cJAczdDIzGVGLBn1"

def get_distance(city1, city2, retries=3):
    """Fetch real road distance (km) using TomTom API with retries."""
    if city1["name"] == city2["name"]:
        return 0.0
    sorted_names = sorted([city1["name"], city2["name"]])
    key = '_'.join(sorted_names)
    if key in distance_cache:
        return distance_cache[key]
    
    print(f"Fetching distance between {city1['name']} and {city2['name']}...")
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{city1['lat']},{city1['lon']}:{city2['lat']},{city2['lon']}/json?routeType=fastest&key={TOMTOM_API_KEY}"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            distance_m = data["routes"][0]["summary"]["lengthInMeters"]
            distance_km = distance_m / 1000.0
            distance_cache[key] = distance_km
            return distance_km
        except Exception as e:
            print(f"Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(1)
            else:
                print(f"Error fetching distance {city1['name']} to {city2['name']}: {e}. Using fallback.")
                dist = math.sqrt((city1['lat'] - city2['lat'])**2 + (city1['lon'] - city2['lon'])**2) * 111
                distance_cache[key] = dist
                return dist

# Build graph as complete without pre-fetching weights
city_names = [city["name"] for city in cities]
G = nx.complete_graph(city_names)
pos_dict = {city["name"]: (city["lon"], city["lat"]) for city in cities}
nx.set_node_attributes(G, pos_dict, "pos")

# Weight function for NX algorithms
def weight_func(u, v, d):
    city_u = next(c for c in cities if c["name"] == u)
    city_v = next(c for c in cities if c["name"] == v)
    return get_distance(city_u, city_v)

# Shortest Path using NX Dijkstra with lazy weight
def get_shortest_path(start, end):
    try:
        path = nx.dijkstra_path(G, start, end, weight=weight_func)
        cost = nx.dijkstra_path_length(G, start, end, weight=weight_func)
        return cost, path
    except nx.NetworkXNoPath:
        return float("inf"), []

# Bellman-Ford implementation
def bellman_ford(graph, start, end):
    nodes = list(graph.nodes)
    dist = {node: float("inf") for node in nodes}
    dist[start] = 0
    pred = {node: None for node in nodes}
    for _ in range(len(nodes) - 1):
        for u, v, data in graph.edges(data=True):
            weight = get_distance(next(c for c in cities if c["name"] == u), 
                                  next(c for c in cities if c["name"] == v))
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                pred[v] = u
            if dist[v] + weight < dist[u]:  # Undirected
                dist[u] = dist[v] + weight
                pred[u] = v
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = pred[current]
    path.reverse()
    if path[0] != start:
        return float("inf"), []
    return dist[end], path

n = len(cities)

# Pre-build distance matrix
print("Initializing distance matrix...")
dist_matrix = [[0] * n for _ in range(n)]
total = n * (n - 1) // 2
count = 0
for i in range(n):
    for j in range(i + 1, n):
        dist = get_distance(cities[i], cities[j])
        dist_matrix[i][j] = dist_matrix[j][i] = dist
        count += 1
        print(f"Processed {count}/{total} distance pairs")
try:
    with open(cache_file, 'w') as f:
        json.dump(distance_cache, f)
except Exception as e:
    print(f"Error saving cache: {e}")
print("Distance matrix initialized.")

# TSP: Held-Karp DP with optional max_budget constraint, start_idx, and subset of cities
def held_karp(dist_matrix, start_idx=0, max_budget=None, subset=None):
    if subset is None:
        subset = list(range(n))
    n_local = len(subset)
    city_map = {subset[i]: i for i in range(n_local)}
    start_idx = city_map[start_idx]
    INF = float("inf")
    dp = [[INF] * n_local for _ in range(1 << n_local)]
    pred = [[-1] * n_local for _ in range(1 << n_local)]
    dp[1 << start_idx][start_idx] = 0
    
    for mask in range(1 << n_local):
        for u in range(n_local):
            if not (mask & (1 << u)) or dp[mask][u] == INF:
                continue
            for v in range(n_local):
                if mask & (1 << v):
                    continue
                new_cost = dp[mask][u] + dist_matrix[subset[u]][subset[v]]
                if max_budget and new_cost > max_budget:
                    continue
                new_mask = mask | (1 << v)
                if new_cost < dp[new_mask][v]:
                    dp[new_mask][v] = new_cost
                    pred[new_mask][v] = u
    
    min_cost = INF
    end = -1
    full_mask = (1 << n_local) - 1
    for u in range(n_local):
        if u == start_idx:
            continue
        cost = dp[full_mask][u] + dist_matrix[subset[u]][subset[start_idx]]
        if cost < min_cost:
            min_cost = cost
            end = u
    if min_cost == INF:
        return INF, []
    
    path = []
    mask = full_mask
    current = end
    while current != -1:
        path.append(current)
        prev = pred[mask][current]
        mask ^= (1 << current)
        current = prev
    path.reverse()
    path.append(start_idx)
    path = [subset[i] for i in path]
    path = [cities[i]["name"] for i in path]
    return min_cost, path

# Heuristic TSP: Nearest Neighbor
def nearest_neighbor(dist_matrix, start=0):
    n_local = len(dist_matrix)
    visited = [False] * n_local
    visited[start] = True
    path = [start]
    cost = 0
    current = start
    for _ in range(n_local - 1):
        min_dist = float("inf")
        next_city = -1
        for v in range(n_local):
            if not visited[v] and dist_matrix[current][v] < min_dist:
                min_dist = dist_matrix[current][v]
                next_city = v
        if next_city == -1:
            break
        visited[next_city] = True
        path.append(next_city)
        cost += min_dist
        current = next_city
    cost += dist_matrix[current][start]
    path.append(start)
    path = [cities[i]["name"] for i in path]
    return cost, path

# Visualization with Matplotlib
def visualize_graph(graph, route=None):
    pos = nx.get_node_attributes(graph, "pos")
    plt.figure(figsize=(12, 10))
    nx.draw(graph, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=8)
    edge_labels = {}
    for u, v in graph.edges():
        weight = get_distance(next(c for c in cities if c["name"] == u), 
                             next(c for c in cities if c["name"] == v))
        edge_labels[(u, v)] = f"{weight:.0f}"
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6)
    if route:
        route_edges = list(zip(route, route[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=route_edges, edge_color="red", width=3)
        nx.draw_networkx_nodes(graph, pos, nodelist=route, node_color="red", node_size=600)
    plt.title("Zimbabwe Cities Graph and Route (Red = Route)")
    plt.show()

# Fetch full route points from TomTom API in Python pairwise
def get_full_route_points(route):
    if len(route) < 2:
        return []
    full_points = []
    for i in range(len(route) - 1):  # For cycle, includes last to first
        city1 = next(c for c in cities if c["name"] == route[i])
        city2 = next(c for c in cities if c["name"] == route[i + 1])
        url = f"https://api.tomtom.com/routing/1/calculateRoute/{city1['lat']},{city1['lon']}:{city2['lat']},{city2['lon']}/json?routeType=fastest&key={TOMTOM_API_KEY}"
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                segment_points = [[p['longitude'], p['latitude']] for p in data['routes'][0]['legs'][0]['points']]
                full_points.extend(segment_points[:-1])  # Avoid duplicate at connection
                full_points.append([city2['lon'], city2['lat']])
                break
            except Exception as e:
                print(f"Attempt {attempt+1}/3 for segment {route[i]} to {route[i+1]}: {e}")
                if attempt == 2:
                    print(f"Failed segment {route[i]} to {route[i+1]}. Using straight line.")
                    full_points.append([city1['lon'], city1['lat']])
                    full_points.append([city2['lon'], city2['lat']])
    return full_points

# Generate TomTom Map HTML
def generate_tomtom_map(route, filename='tomtom_map.html'):
    if not route or len(route) < 2:
        print("No valid route to visualize.")
        return
    full_points = get_full_route_points(route)
    city_coords = [[c["lon"], c["lat"]] for c in cities if c["name"] in route]
    
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": full_points
            },
            "properties": {}
        }]
    }
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>TomTom Map - Zimbabwe Route</title>
        <script src="https://api.tomtom.com/maps-sdk-for-web/cdn/6.x/6.25.0/maps/maps-web.min.js"></script>
        <style>
            #map {{ height: 100vh; width: 100vw; margin: 0; }}
            body, html {{ margin: 0; padding: 0; height: 100vh; overflow: hidden; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            console.log('Starting map initialization...');
            var apiKey = '{TOMTOM_API_KEY}';
            tt.setProductInfo('Zimbabwe Route Finder', '1.0');
            var map = tt.map({{
                key: apiKey,
                container: 'map',
                center: [29.0, -19.0],
                zoom: 6
            }});
            map.on('load', function() {{
                console.log('Map loaded');
                var routeCities = {json.dumps(route)};
                var cityCoords = {json.dumps(city_coords)};
                var geojson = {json.dumps(geojson)};
                console.log('GeoJSON:', geojson);
                console.log('City coords:', cityCoords);
                console.log('Route cities:', routeCities);
                map.addLayer({{
                    'id': 'route',
                    'type': 'line',
                    'source': {{
                        'type': 'geojson',
                        'data': geojson
                    }},
                    'paint': {{
                        'line-color': 'red',
                        'line-width': 4
                    }}
                }});
                console.log('Route layer added');
                var bounds = new tt.LngLatBounds();
                geojson.features[0].geometry.coordinates.forEach(function(point) {{
                    bounds.extend(point);
                }});
                map.fitBounds(bounds, {{ padding: 50 }});
                console.log('Bounds set');
                var startCoord = cityCoords[0];
                var endCoord = cityCoords[cityCoords.length - 1];
                var startPopup = new tt.Popup({{offset: 25}}).setHTML('<strong>Start: ' + routeCities[0] + '</strong>');
                new tt.Marker({{color: 'green'}}).setLngLat(startCoord).setPopup(startPopup).addTo(map);
                console.log('Start marker added for', routeCities[0]);
                var endPopup = new tt.Popup({{offset: 25}}).setHTML('<strong>End: ' + routeCities[routeCities.length - 1] + '</strong>');
                new tt.Marker({{color: 'blue'}}).setLngLat(endCoord).setPopup(endPopup).addTo(map);
                console.log('End marker added for', routeCities[routeCities.length - 1]);
                for (var index = 1; index < cityCoords.length - 1; index++) {{
                    var coord = cityCoords[index];
                    var popup = new tt.Popup({{offset: 25}}).setHTML('<strong>' + routeCities[index] + '</strong>');
                    new tt.Marker().setLngLat(coord).setPopup(popup).addTo(map);
                    console.log('Marker added for', routeCities[index]);
                }}
                // Auto-refresh mechanism
                setInterval(function() {{
                    fetch('/check_update?t=' + new Date().getTime()).then(response => response.json()).then(data => {{
                        if (data.newRoute) {{
                            window.location.reload();
                        }}
                    }}).catch(err => console.error('Check update failed:', err));
                }}, 3000);
            }});
            map.on('error', function(error) {{
                console.error('Map error:', error);
            }});
        </script>
    </body>
    </html>
    """
    try:
        with open(filename, 'w') as f:
            f.write(html_template)
        print(f"TomTom map updated: {filename}. Refresh browser at http://localhost:8000/tomtom_map.html")
    except Exception as e:
        print(f"Error generating TomTom map: {e}")

# Local server with update check
class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/check_update'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            global last_route_hash
            current_hash = hash(tuple(last_route)) if last_route else 0
            response = {'newRoute': current_hash != last_route_hash}
            self.wfile.write(json.dumps(response).encode())
            if response['newRoute']:
                last_route_hash = current_hash
        else:
            super().do_GET()

last_route = None
last_route_hash = 0

# GUI Setup
def create_gui():
    global last_route, last_route_hash
    root = tk.Tk()
    root.title("Zimbabwe Optimal Route Finder")
    # Near-fullscreen height (100vh equivalent)
    screen_height = root.winfo_screenheight()
    root.geometry(f"900x{screen_height-100}")
    root.resizable(True, True)

    # Create canvas for background
    canvas = tk.Canvas(root, highlightthickness=0)
    canvas.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Load Zimbabwe flag image (place 'zimbabwe_flag.jpg' in project directory)
    try:
        bg_image = Image.open("zimbabwe_flag.jpg")
        bg_image = bg_image.resize((900, screen_height-100), Image.LANCZOS)
        bg_photo = ImageTk.PhotoImage(bg_image)
        canvas.create_image(0, 0, image=bg_photo, anchor="nw")
        # Semi-transparent overlay for readability
        canvas.create_rectangle(0, 0, 900, screen_height-100, fill="white", stipple="gray50")
    except Exception as e:
        print(f"Error loading background image: {e}. Using plain background.")
        canvas.configure(bg="#2E7D32")  # Zimbabwe flag green

    # Main frame with padding
    main_frame = ttk.Frame(canvas, padding="10", style="Main.TFrame")
    main_frame.grid(row=0, column=0, sticky="nsew")
    canvas.create_window(450, screen_height/2-50, window=main_frame, anchor="center")

    # Style configuration
    style = ttk.Style()
    style.configure("Main.TFrame", background="white")
    style.configure("TButton", font=("Arial", 10), padding=5, background="#FFCA28")  # Zimbabwe flag yellow
    style.configure("TLabel", font=("Arial", 12), background="white")
    style.configure("TCombobox", font=("Arial", 10))
    style.configure("TEntry", font=("Arial", 10))

    # Title
    ttk.Label(main_frame, text="Zimbabwe Route Planner", font=("Arial", 16, "bold"), foreground="#D32F2F").grid(row=0, column=0, columnspan=2, pady=10)  # Red from flag

    # Input frame
    input_frame = ttk.LabelFrame(main_frame, text="Route Selection", padding=10)
    input_frame.grid(row=1, column=0, sticky="ew", pady=5)
    
    ttk.Label(input_frame, text="Start City:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    start_combo = ttk.Combobox(input_frame, values=city_names, width=20)
    start_combo.grid(row=0, column=1, padx=10, pady=5, sticky="w")
    start_combo.current(0)

    ttk.Label(input_frame, text="End City:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
    end_combo = ttk.Combobox(input_frame, values=city_names, width=20)
    end_combo.grid(row=1, column=1, padx=10, pady=5, sticky="w")
    end_combo.current(1)

    # Constraints frame
    constraints_frame = ttk.LabelFrame(main_frame, text="Constraints", padding=10)
    constraints_frame.grid(row=2, column=0, sticky="ew", pady=5)
    
    ttk.Label(constraints_frame, text="Max Budget (km):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    max_budget_entry = ttk.Entry(constraints_frame, width=10)
    max_budget_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")
    max_budget_entry.insert(0, "5000")

    ttk.Label(constraints_frame, text="Mandatory Stops:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
    mandatory_listbox = tk.Listbox(constraints_frame, selectmode=tk.MULTIPLE, height=6, font=("Arial", 10))
    mandatory_listbox.grid(row=1, column=1, padx=10, pady=5, sticky="w")
    for city in city_names:
        mandatory_listbox.insert(tk.END, city)

    # Buttons frame
    buttons_frame = ttk.LabelFrame(main_frame, text="Actions", padding=10)
    buttons_frame.grid(row=3, column=0, sticky="ew", pady=5)
    
    # Define functions before button assignments
    def compute_shortest():
        global last_route, last_route_hash
        start = start_combo.get()
        end = end_combo.get()
        if start not in city_names or end not in city_names:
            result_text.insert(tk.END, "Invalid cities selected.\n")
            return
        try:
            max_budget = float(max_budget_entry.get() or float('inf'))
        except ValueError:
            result_text.insert(tk.END, "Invalid max budget value.\n")
            return
        mandatory_selected = [mandatory_listbox.get(i) for i in mandatory_listbox.curselection()]
        if mandatory_selected:
            path = [start]
            cost = 0
            current = start
            for mandatory in mandatory_selected + [end]:
                segment_cost, segment_path = get_shortest_path(current, mandatory)
                if segment_cost == float("inf"):
                    result_text.insert(tk.END, f"No path from {current} to {mandatory}.\n")
                    return
                path.extend(segment_path[1:])
                cost += segment_cost
                current = mandatory
            result_text.delete(1.0, tk.END)
            if cost > max_budget:
                result_text.insert(tk.END, f"Shortest Path with Mandatory Stops: Cost {cost:.2f} km EXCEEDS max budget of {max_budget:.2f} km!\nPath: {path}\n\n")
            else:
                result_text.insert(tk.END, f"Shortest Path with Mandatory Stops: Cost {cost:.2f} km\nPath: {path}\n\n")
        else:
            cost, path = get_shortest_path(start, end)
            result_text.delete(1.0, tk.END)
            if cost == float("inf"):
                result_text.insert(tk.END, f"No path from {start} to {end}.\n")
                return
            if cost > max_budget:
                result_text.insert(tk.END, f"Shortest Path (Dijkstra): {start} to {end}\nCost: {cost:.2f} km EXCEEDS max budget of {max_budget:.2f} km!\nPath: {path}\n\n")
            else:
                result_text.insert(tk.END, f"Shortest Path (Dijkstra): {start} to {end}\nCost: {cost:.2f} km\nPath: {path}\n\n")
        last_route = path
        generate_tomtom_map(path)

    def compute_bellman_ford():
        global last_route, last_route_hash
        start = start_combo.get()
        end = end_combo.get()
        if start not in city_names or end not in city_names:
            result_text.insert(tk.END, "Invalid cities selected.\n")
            return
        try:
            max_budget = float(max_budget_entry.get() or float('inf'))
        except ValueError:
            result_text.insert(tk.END, "Invalid max budget value.\n")
            return
        mandatory_selected = [mandatory_listbox.get(i) for i in mandatory_listbox.curselection()]
        if mandatory_selected:
            path = [start]
            cost = 0
            current = start
            for mandatory in mandatory_selected + [end]:
                segment_cost, segment_path = bellman_ford(G, current, mandatory)
                if segment_cost == float("inf"):
                    result_text.insert(tk.END, f"No path from {current} to {mandatory}.\n")
                    return
                path.extend(segment_path[1:])
                cost += segment_cost
                current = mandatory
            result_text.delete(1.0, tk.END)
            if cost > max_budget:
                result_text.insert(tk.END, f"Bellman-Ford Path with Mandatory Stops: Cost {cost:.2f} km EXCEEDS max budget of {max_budget:.2f} km!\nPath: {path}\n\n")
            else:
                result_text.insert(tk.END, f"Bellman-Ford Path with Mandatory Stops: Cost {cost:.2f} km\nPath: {path}\n\n")
        else:
            cost, path = bellman_ford(G, start, end)
            result_text.delete(1.0, tk.END)
            if cost == float("inf"):
                result_text.insert(tk.END, f"No path from {start} to {end}.\n")
                return
            if cost > max_budget:
                result_text.insert(tk.END, f"Bellman-Ford Path: {start} to {end}\nCost: {cost:.2f} km EXCEEDS max budget of {max_budget:.2f} km!\nPath: {path}\n\n")
            else:
                result_text.insert(tk.END, f"Bellman-Ford Path: {start} to {end}\nCost: {cost:.2f} km\nPath: {path}\n\n")
        last_route = path
        generate_tomtom_map(path)

    def compute_tsp():
        global last_route, last_route_hash
        result_text.delete(1.0, tk.END)
        start_name = start_combo.get()
        if start_name not in city_names:
            result_text.insert(tk.END, "Invalid start city.\n")
            return
        start_idx = city_names.index(start_name)
        try:
            max_budget = float(max_budget_entry.get() or float('inf'))
        except ValueError:
            result_text.insert(tk.END, "Invalid max budget value.\n")
            return
        mandatory_selected = [mandatory_listbox.get(i) for i in mandatory_listbox.curselection()]
        subset_indices = [city_names.index(start_name)]
        for mandatory in mandatory_selected:
            if mandatory not in city_names:
                result_text.insert(tk.END, "Invalid mandatory stop.\n")
                return
            subset_indices.append(city_names.index(mandatory))
        subset_indices = list(set(subset_indices))  # Remove duplicates
        if len(subset_indices) > 1:
            cost, path = held_karp(dist_matrix, city_names.index(start_name), max_budget, subset_indices)
        else:
            cost, path = held_karp(dist_matrix, start_idx, max_budget)
        result_text.insert(tk.END, f"TSP Cost: {cost:.2f} km\nPath: {path}\n\n")
        last_route = path
        generate_tomtom_map(path)

    def visualize_matplotlib():
        global last_route
        if last_route:
            visualize_graph(G, last_route)
        else:
            result_text.insert(tk.END, "Compute a path or TSP first.\n")

    def visualize_tomtom():
        global last_route
        if last_route:
            generate_tomtom_map(last_route)
            webbrowser.open(f'http://localhost:8000/tomtom_map.html?t={int(time.time())}', new=0, autoraise=True)
            result_text.insert(tk.END, "TomTom map opened/updated in browser.\n")
        else:
            result_text.insert(tk.END, "Compute a path or TSP first.\n")

    def scalability_analysis():
        result_text.insert(tk.END, "Running scalability analysis...\n")
        root.update()
        start_time = time.time()
        held_karp(dist_matrix, 0, float('inf'))
        time_taken = time.time() - start_time
        space_approx = n * (1 << n) * 8 * 2  # dp and pred, approximate bytes
        result_text.insert(tk.END, f"Full graph (n={n}): Time {time_taken:.2f}s, Approx Space {space_approx / 1024:.2f} KB\n")
        if n > 12:
            result_text.insert(tk.END, "For n>12, uses heuristic for scalability.\n")
        result_text.insert(tk.END, "Complexity: O(n^2 * 2^n) time, O(n * 2^n) space for Held-Karp.\n\n")

    def clear_cache():
        global distance_cache, dist_matrix
        distance_cache = {}
        if os.path.exists(cache_file):
            os.remove(cache_file)
        result_text.insert(tk.END, "Cache cleared. Rebuilding distance matrix...\n")
        root.update()
        dist_matrix = [[0] * n for _ in range(n)]
        total = n * (n - 1) // 2
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = get_distance(cities[i], cities[j])
                dist_matrix[i][j] = dist_matrix[j][i] = dist
                count += 1
                progress_var.set(count / total * 100)
                print(f"Processed {count}/{total} distance pairs")
                root.update()
        try:
            with open(cache_file, 'w') as f:
                json.dump(distance_cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
        result_text.insert(tk.END, "Distance matrix rebuilt.\n")

    # Button assignments
    ttk.Button(buttons_frame, text="Compute Shortest Path (Dijkstra)", command=compute_shortest).grid(row=0, column=0, padx=5, pady=5)
    ttk.Button(buttons_frame, text="Compute Shortest Path (Bellman-Ford)", command=compute_bellman_ford).grid(row=0, column=1, padx=5, pady=5)
    ttk.Button(buttons_frame, text="Compute TSP (All Cities)", command=compute_tsp).grid(row=0, column=2, padx=5, pady=5)
    ttk.Button(buttons_frame, text="Visualize Graph/Route (Matplotlib)", command=visualize_matplotlib).grid(row=1, column=0, padx=5, pady=5)
    ttk.Button(buttons_frame, text="View TomTom Map", command=visualize_tomtom).grid(row=1, column=1, padx=5, pady=5)
    ttk.Button(buttons_frame, text="Scalability Analysis", command=scalability_analysis).grid(row=1, column=2, padx=5, pady=5)
    ttk.Button(buttons_frame, text="Clear Cache", command=clear_cache).grid(row=1, column=3, padx=5, pady=5)

    # Result text area
    result_text = tk.Text(main_frame, height=8, width=50, font=("Arial", 10), bg="#FFFFFF", fg="#000000")
    result_text.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")

    # Progress bar
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(main_frame, variable=progress_var, maximum=100)
    progress_bar.grid(row=5, column=0, padx=10, pady=5, sticky="ew")

    # Note about map
    ttk.Label(main_frame, text="View map at http://localhost:8000/tomtom_map.html", font=("Arial", 10, "italic"), foreground="#000000").grid(row=6, column=0, padx=10, pady=5)

    # Configure grid weights
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(4, weight=1)

    # Keep background image reference to prevent garbage collection
    root.bg_photo = bg_photo

    # Start local server
    def start_server():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        Handler = CustomHandler
        with socketserver.TCPServer(("", 8000), Handler) as httpd:
            print("Local server running at http://localhost:8000")
            httpd.serve_forever()

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Initial map
    generate_tomtom_map([])

    root.mainloop()

# Main Execution: Run GUI
if __name__ == "__main__":
    create_gui()