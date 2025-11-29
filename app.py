# app.py
import streamlit as st
import random
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from itertools import product
from collections import defaultdict

st.set_page_config(page_title="3D NoC Simulator — Enhanced", layout="wide")

# -------------------------
# Basic Data Structures
# -------------------------
class RouterNode:
    def __init__(self, x, y, z, vc_count=2, buffer_size=4):
        self.x, self.y, self.z = x, y, z
        self.buffer = []
        self.buffer_size = buffer_size
        self.vc_count = vc_count
        self.vc_usage = [0] * vc_count

    def has_space(self):
        return len(self.buffer) < self.buffer_size

    def push_packet(self, packet):
        if self.has_space():
            self.buffer.append(packet)
            return True
        return False

    def pop_packet(self):
        if self.buffer:
            pkt = self.buffer.pop(0)
            # simple VC selection: choose least used VC
            vc = int(np.argmin(self.vc_usage))
            self.vc_usage[vc] += 1
            return pkt
        return None

class Packet:
    def __init__(self, src, dst, born_step=0):
        self.src = src
        self.dst = dst
        self.hops = 0
        self.finished = False
        self.born_step = born_step
        self.delivered_step = None

# -------------------------
# Routing & Faults
# -------------------------
def dimde_route(src, dst):
    sx, sy, sz = src
    dx, dy, dz = dst
    # X -> Y -> Z (Dimensional decomposition)
    if sx != dx: return (int((dx - sx) / abs(dx - sx)), 0, 0)
    if sy != dy: return (0, int((dy - sy) / abs(dy - sy)), 0)
    if sz != dz: return (0, 0, int((dz - sz) / abs(dz - sz)))
    return (0, 0, 0)

def is_link_faulty(prob):
    return random.random() < prob

# -------------------------
# Simulation Step
# -------------------------
def simulate_step(grid, size, step_num, inj_rate, fault_prob, energy_hop, logs, fault_edges, energy_layer):
    nodes = [(x, y, z) for x, y, z in product(range(size), repeat=3)]

    # Traffic injection (procedural, no dataset)
    for (x, y, z) in nodes:
        if random.random() < inj_rate:
            dst = random.choice(nodes)
            if dst != (x, y, z):
                grid[x][y][z].push_packet(Packet((x, y, z), dst, born_step=step_num))
                logs.append({
                    "step": step_num, "event": "inject", "node": (x, y, z), "dst": dst
                })

    delivered = 0
    energy_consumed = 0

    for (x, y, z) in nodes:
        node = grid[x][y][z]
        # record buffer snapshot pre-step
        logs.append({"step": step_num, "event": "buffer_snapshot", "node": (x, y, z), "buffer_len": len(node.buffer)})

        if node.buffer:
            pkt = node.pop_packet()
            if not pkt:
                continue

            # decide direction
            dx, dy, dz = dimde_route((x, y, z), pkt.dst)

            # if at destination
            if dx == dy == dz == 0:
                pkt.finished = True
                pkt.delivered_step = step_num
                delivered += 1
                energy_consumed += pkt.hops * energy_hop
                energy_layer[pkt.dst[2]] += pkt.hops * energy_hop
                logs.append({"step": step_num, "event": "deliver", "node": (x, y, z), "pkt_src": pkt.src, "pkt_dst": pkt.dst, "hops": pkt.hops})
                continue

            # fault injection on the link attempt
            link = ((x, y, z), (x + dx, y + dy, z + dz))
            if is_link_faulty(fault_prob):
                # mark this link as faulted for visualization and reroute attempt
                fault_edges.add(normalize_edge(link))
                logs.append({"step": step_num, "event": "fault", "from": link[0], "to": link[1]})
                # attempt minimal alternate: try other remaining dims
                if dx != 0:
                    # try Y first then Z
                    if y != pkt.dst[1]:
                        dx, dy, dz = 0, int((pkt.dst[1] - y) / abs(pkt.dst[1] - y)), 0
                    elif z != pkt.dst[2]:
                        dx, dy, dz = 0, 0, int((pkt.dst[2] - z) / abs(pkt.dst[2] - z))
                    else:
                        dx, dy, dz = 0, 0, 0
                elif dy != 0:
                    if z != pkt.dst[2]:
                        dx, dy, dz = 0, 0, int((pkt.dst[2] - z) / abs(pkt.dst[2] - z))
                    else:
                        dx, dy, dz = 0, 0, 0
                else:
                    dx, dy, dz = 0, 0, 0

            nx, ny, nz = x + dx, y + dy, z + dz
            # valid move and in-range?
            if 0 <= nx < size and 0 <= ny < size and 0 <= nz < size:
                neigh = grid[nx][ny][nz]
                if neigh.has_space():
                    pkt.hops += 1
                    neigh.push_packet(pkt)
                    logs.append({"step": step_num, "event": "forward", "from": (x, y, z), "to": (nx, ny, nz), "hops": pkt.hops})
                else:
                    # blocked — pushback (for simplicity: drop or keep; here we requeue at source node if space)
                    if node.has_space():
                        node.push_packet(pkt)
                        logs.append({"step": step_num, "event": "blocked_requeue", "node": (x, y, z)})
                    else:
                        # drop packet
                        logs.append({"step": step_num, "event": "drop", "node": (x, y, z)})
            else:
                # out-of-range (should not happen) — drop
                logs.append({"step": step_num, "event": "drop_oob", "node": (x, y, z)})

    # return per-step summary
    return {
        "delivered": delivered,
        "energy": energy_consumed,
        "buffers": { (x,y,z): len(grid[x][y][z].buffer) for x,y,z in nodes },
        "vc_usage": { (x,y,z): list(grid[x][y][z].vc_usage) for x,y,z in nodes }
    }

def normalize_edge(edge):
    a, b = edge
    return tuple(sorted([a, b]))

# -------------------------
# Streamlit UI
# -------------------------
st.title("🔷 Enhanced 3D NoC Simulator — Visualization + Logs")
st.markdown("Interactive simulator demonstrating DimDe routing, dynamic VC usage, faults, and energy breakdown. No external datasets or models used.")

# Controls
left, right = st.columns([1, 2])
with left:
    size = st.slider("3D NoC Dimension (NxNxN)", 2, 6, 3)
    vc_count = st.slider("Virtual Channels per Router", 1, 8, 2)
    buffer_size = st.slider("Buffer Size per Router", 1, 16, 4)
    energy_hop = st.slider("Energy per Hop (pJ)", 1, 20, 5)
    inj_rate = st.slider("Packet Injection Rate (prob per node/step)", 0.0, 1.0, 0.15)
    fault_prob = st.slider("Fault Probability per attempted link", 0.0, 0.5, 0.08)
    steps = st.slider("Simulation Steps", 1, 200, 40)
    rng_seed = st.number_input("Random seed (0 = random)", 0, 999999, 0)
    run_button = st.button("Run Simulation")

with right:
    st.markdown("**Quick tips:** adjust dimensions and steps to see richer behavior. Faults are recorded and visualized.")

# initialize grid if run
if rng_seed != 0:
    random.seed(rng_seed)
    np.random.seed(rng_seed)

if run_button:
    # build grid
    grid = [[[RouterNode(x, y, z, vc_count, buffer_size) for z in range(size)]
              for y in range(size)]
             for x in range(size)]

    logs = []
    fault_edges = set()
    energy_layer = [0.0 for _ in range(size)]
    summary_steps = []
    progress = st.progress(0)

    for step in range(1, steps + 1):
        res = simulate_step(grid, size, step, inj_rate, fault_prob, energy_hop, logs, fault_edges, energy_layer)
        summary_steps.append({
            "step": step,
            "delivered": res["delivered"],
            "energy": res["energy"],
            "avg_buffer": np.mean(list(res["buffers"].values()))
        })
        progress.progress(step / steps)

    progress.empty()
    st.success(f"Simulation finished: {sum(s['delivered'] for s in summary_steps)} packets delivered over {steps} steps.")

    # -------------------------
    # Prepare visualizations
    # -------------------------
    # 1) 3D topology scatter + edges (show current buffer congestion at end)
    xs, ys, zs, sizesc, colors = [], [], [], [], []
    nodes = [(x, y, z) for x, y, z in product(range(size), repeat=3)]
    buffer_map = { (x,y,z): len(grid[x][y][z].buffer) for x,y,z in nodes }
    max_buf = max(buffer_map.values()) if buffer_map else 1

    for (x,y,z) in nodes:
        xs.append(x)
        ys.append(y)
        zs.append(z)
        sizesc.append(8 + (buffer_map[(x,y,z)] / max_buf) * 24)
        colors.append(buffer_map[(x,y,z)])

    edge_x, edge_y, edge_z = [], [], []
    for (x, y, z) in nodes:
        for dx, dy, dz in [(1,0,0),(0,1,0),(0,0,1)]:  # positive directions only to avoid duplicates
            nx, ny, nz = x+dx, y+dy, z+dz
            if 0 <= nx < size and 0 <= ny < size and 0 <= nz < size:
                # draw a line between (x,y,z) and (nx,ny,nz)
                edge_x += [x, nx, None]
                edge_y += [y, ny, None]
                edge_z += [z, nz, None]

    fig = go.Figure()
    # edges
    fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines',
                               line=dict(width=2, color='lightgray'),
                               hoverinfo='none', name='links'))
    # nodes
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='markers',
                               marker=dict(size=sizesc, color=colors, colorscale='YlOrRd', showscale=True),
                               text=[f"node={n}<br>buffer={buffer_map[n]}" for n in nodes],
                               hoverinfo='text', name='routers'))
    fig.update_layout(scene=dict(aspectmode='cube'), margin=dict(l=0,r=0,t=30,b=0),
                      title="3D NoC Topology — node color = buffer occupancy")
    st.plotly_chart(fig, use_container_width=True)

    # 2) Fault map overlay (list faults)
    if fault_edges:
        st.subheader("Fault Map (links that experienced faults during the run)")
        fault_list = [{"from": e[0], "to": e[1]} for e in fault_edges]
        st.table(pd.DataFrame(fault_list))

        # show fault edges in 3D (red lines)
        fx, fy, fz = [], [], []
        for a, b in fault_edges:
            (x1,y1,z1), (x2,y2,z2) = a, b
            fx += [x1, x2, None]
            fy += [y1, y2, None]
            fz += [z1, z2, None]
        fig_fault = go.Figure()
        fig_fault.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(width=2, color='lightgray'), hoverinfo='none'))
        fig_fault.add_trace(go.Scatter3d(x=fx, y=fy, z=fz, mode='lines', line=dict(width=6, color='red'), name='faults'))
        fig_fault.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='markers',
                                         marker=dict(size=sizesc, color=colors, colorscale='YlOrRd', showscale=False)))
        fig_fault.update_layout(scene=dict(aspectmode='cube'), margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_fault, use_container_width=True)

    # 3) VC usage heat / distribution at end
    st.subheader("VC Usage Snapshot (per-node, list of VC counts)")
    vc_rows = []
    for (x,y,z) in nodes:
        vc_rows.append({"node": (x,y,z), "vc_usage": grid[x][y][z].vc_usage})
    df_vc = pd.DataFrame(vc_rows)
    st.dataframe(df_vc)

    # 4) Energy per layer bar chart
    st.subheader("Energy Consumption per Layer (pJ)")
    df_energy = pd.DataFrame({"layer": list(range(size)), "energy_pJ": energy_layer})
    st.bar_chart(df_energy.set_index("layer"))

    # 5) Time-series summaries (delivered, avg buffer)
    st.subheader("Per-step summary")
    df_summary = pd.DataFrame(summary_steps)
    st.line_chart(df_summary.set_index("step")[["delivered", "avg_buffer"]])

    # 6) Logs and download
    st.subheader("Event Logs (sample of last 200 entries)")
    df_logs = pd.DataFrame(logs)
    st.dataframe(df_logs.tail(200))

    csv = df_logs.to_csv(index=False)
    st.download_button("Download full logs (CSV)", csv, file_name="noc_simulation_logs.csv", mime="text/csv")
