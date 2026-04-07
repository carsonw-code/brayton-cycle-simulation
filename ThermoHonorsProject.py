# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:48:38 2026

@author: carso
"""

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider 
from CoolProp.CoolProp import PropsSI 

FLUID = "Air"

# Simple helper functions

def state_from_PT(P, T):
    # Return a state when pressure and temperature are known.
    
    return {                                        
        "P": P,
        "T": T,
        "h": PropsSI("H", "P", P, "T", T, FLUID),
        "s": PropsSI("S", "P", P, "T", T, FLUID),
    }

def state_from_PS(P, s):
    # Return a state when pressure and entropy are known.
    return {
        "P": P,
        "T": PropsSI("T", "P", P, "S", s, FLUID),
        "h": PropsSI("H", "P", P, "S", s, FLUID),
        "s": s,
    }

def state_from_PH(P, h):
    # Return a state when pressure and enthalpy are known.
    return {
        "P": P,
        "T": PropsSI("T", "P", P, "H", h, FLUID),
        "h": h,
        "s": PropsSI("S", "P", P, "H", h, FLUID),
    }

def constant_pressure_curve(P, T_start, T_end, points=20):
    # Create points for a constant-pressure line on a T-s diagram.
    T_values = np.linspace(T_start, T_end, points)
    s_values = []

    for T in T_values:
        s = PropsSI("S", "P", P, "T", T, FLUID) / 1000
        s_values.append(s)

    return np.array(s_values), T_values


# Main Brayton cycle calculation

def calculate_cycle(T1, P1_kPa, rp, T3, eta_c, eta_t):
    P1 = P1_kPa * 1000
    P2 = P1 * rp

    # State 1: compressor inlet
    state1 = state_from_PT(P1, T1)

    # State 2s: ideal compressor exit
    state2s = state_from_PS(P2, state1["s"])

    # State 2: real compressor exit using compressor efficiency
    h2 = state1["h"] + (state2s["h"] - state1["h"]) / eta_c
    state2 = state_from_PH(P2, h2)

    # State 3: turbine inlet after heat addition
    state3 = state_from_PT(P2, T3)

    # State 4s: ideal turbine exit
    state4s = state_from_PS(P1, state3["s"])

    # State 4: real turbine exit using turbine efficiency
    h4 = state3["h"] - eta_t * (state3["h"] - state4s["h"])
    state4 = state_from_PH(P1, h4)

    # Process lines for plotting
    s12 = np.array([state1["s"], state2["s"]]) / 1000
    T12 = np.array([state1["T"], state2["T"]])

    s23, T23 = constant_pressure_curve(P2, state2["T"], state3["T"])

    s34 = np.array([state3["s"], state4["s"]]) / 1000
    T34 = np.array([state3["T"], state4["T"]])

    s41, T41 = constant_pressure_curve(P1, state4["T"], state1["T"])

    # Performance values
    w_c = state2["h"] - state1["h"]
    w_t = state3["h"] - state4["h"]
    q_in = state3["h"] - state2["h"]
    q_out = state4["h"] - state1["h"]
    w_net = w_t - w_c
    eta_th = w_net / q_in

    return {                                             
        "states": [state1, state2, state3, state4],   
        "s12": s12, "T12": T12,
        "s23": s23, "T23": T23,
        "s34": s34, "T34": T34,
        "s41": s41, "T41": T41,
        "w_c": w_c,
        "w_t": w_t,
        "q_in": q_in,
        "q_out": q_out,
        "w_net": w_net,
        "eta_th": eta_th,
        "eta_c": eta_c,
        "eta_t": eta_t
    }


# Initial slider values

T1_start = 300
P1_start = 101.325
rp_start = 8
T3_start = 1400
eta_c_start = 0.85
eta_t_start = 0.90


# Create figure

fig, ax = plt.subplots(figsize=(11.5, 7.8))
plt.subplots_adjust(left=0.09, bottom=0.40, right=0.68)

data = calculate_cycle(T1_start, P1_start, rp_start, T3_start, eta_c_start, eta_t_start)

line12, = ax.plot(data["s12"], data["T12"], lw=2, label="1→2 Compressor")
line23, = ax.plot(data["s23"], data["T23"], lw=2, label="2→3 Heat Addition")
line34, = ax.plot(data["s34"], data["T34"], lw=2, label="3→4 Turbine")
line41, = ax.plot(data["s41"], data["T41"], lw=2, label="4→1 Heat Rejection")

# Plot state points
states = data["states"]
s_points = [state["s"] / 1000 for state in states]
T_points = [state["T"] for state in states]

scatter = ax.scatter(s_points, T_points, zorder=3)

# Add state number labels
point_labels = []
for i in range(4):
    label = ax.text(s_points[i], T_points[i], f"  {i+1}", fontsize=11, weight="bold")
    point_labels.append(label)

# Title and axes
title = ax.set_title(
    rf"Brayton Cycle on $T$-$s$ Diagram    $\eta_{{th}} = {data['eta_th']*100:.2f}\%$"
)
ax.set_xlabel(r"Entropy, $s$ [kJ/kg·K]")
ax.set_ylabel(r"Temperature, $T$ [K]")
ax.grid(True)
ax.legend(loc="upper left")


# Right-side info boxes

variable_box_text = (
    "$T_1$ = compressor inlet temp.\n"
    "$P_1$ = compressor inlet pressure\n"
    "$r_p$ = pressure ratio = $P_2/P_1$\n"
    "$T_3$ = turbine inlet temp.\n"
    "$\\eta_c$ = compressor efficiency\n"
    "$\\eta_t$ = turbine efficiency\n"
    "$\\eta_{th}$ = thermal efficiency\n"
    "$s$ = specific entropy\n"
    "$T$ = temperature"
)

fig.text(
    0.72, 0.88,
    variable_box_text,
    fontsize=9.5,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
)

performance_text = fig.text(
    0.72, 0.50,
    "",
    fontsize=9.5,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
)

net_work_text = fig.text(
    0.72, 0.24,
    "",
    fontsize=13,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.95)
)

def update_performance_text(data):
    performance_text.set_text(
        rf"$\eta_c$ = {data['eta_c']:.2f}" "\n"
        rf"$\eta_t$ = {data['eta_t']:.2f}" "\n"
        rf"$w_c$ = {data['w_c']/1000:.2f} kJ/kg" "\n"
        rf"$w_t$ = {data['w_t']/1000:.2f} kJ/kg" "\n"
        rf"$q_{{in}}$ = {data['q_in']/1000:.2f} kJ/kg" "\n"
        rf"$q_{{out}}$ = {data['q_out']/1000:.2f} kJ/kg" "\n"
        rf"$\eta_{{th}}$ = {data['eta_th']*100:.2f}%"
    )

def update_net_work_text(data):
    net_work_text.set_text(
        "Net Work Output\n"
        rf"$w_{{net}}$ = {data['w_net']/1000:.2f} kJ/kg"
    )

update_performance_text(data)
update_net_work_text(data)


# Sliders

ax_T1   = plt.axes([0.15, 0.28, 0.45, 0.03])
ax_P1   = plt.axes([0.15, 0.23, 0.45, 0.03])
ax_rp   = plt.axes([0.15, 0.18, 0.45, 0.03])
ax_T3   = plt.axes([0.15, 0.13, 0.45, 0.03])
ax_etac = plt.axes([0.15, 0.08, 0.45, 0.03])
ax_etat = plt.axes([0.15, 0.03, 0.45, 0.03])

slider_T1 = Slider(ax_T1, r"$T_1$ [K]", 250, 400, valinit=T1_start)
slider_P1 = Slider(ax_P1, r"$P_1$ [kPa]", 80, 200, valinit=P1_start)
slider_rp = Slider(ax_rp, r"$r_p$", 2, 30, valinit=rp_start)
slider_T3 = Slider(ax_T3, r"$T_3$ [K]", 700, 2200, valinit=T3_start)
slider_etac = Slider(ax_etac, r"$\eta_c$", 0.60, 1.00, valinit=eta_c_start)
slider_etat = Slider(ax_etat, r"$\eta_t$", 0.60, 1.00, valinit=eta_t_start)


# Update function

def update(val):
    T1 = slider_T1.val
    P1 = slider_P1.val
    rp = slider_rp.val
    T3 = slider_T3.val
    eta_c = slider_etac.val
    eta_t = slider_etat.val

    data = calculate_cycle(T1, P1, rp, T3, eta_c, eta_t)

    # Update line data
    line12.set_data(data["s12"], data["T12"])
    line23.set_data(data["s23"], data["T23"])
    line34.set_data(data["s34"], data["T34"])
    line41.set_data(data["s41"], data["T41"])

    # Update state points
    states = data["states"]
    s_points = [state["s"] / 1000 for state in states]
    T_points = [state["T"] for state in states]
    scatter.set_offsets(np.column_stack((s_points, T_points)))

    # Move point labels
    for i in range(4):
        point_labels[i].set_position((s_points[i], T_points[i]))

    # Update title and info boxes
    title.set_text(
        rf"Brayton Cycle on $T$-$s$ Diagram    $\eta_{{th}} = {data['eta_th']*100:.2f}\%$"
    )
    update_performance_text(data)
    update_net_work_text(data)

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

# Connect sliders
slider_T1.on_changed(update)
slider_P1.on_changed(update)
slider_rp.on_changed(update)
slider_T3.on_changed(update)
slider_etac.on_changed(update)
slider_etat.on_changed(update)

plt.show()