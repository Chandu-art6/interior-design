"""
Streamlit UI Components and Rich Visualizations
High-end, sleek architectural Matplotlib charts and utility display functions.
"""
from typing import Dict, Any, List, Optional
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import streamlit as st


def generate_comfort_radar_chart(metrics: Dict[str, float]) -> Image.Image:
    """Generates a clean, modern radar chart evaluating room layout metrics."""
    categories = ['Circulation', 'Lighting', 'Accessibility', 'Acoustics', 'Ergonomics']
    
    values = [
        metrics.get("circulation_score", 85.0),
        metrics.get("lighting_score", 90.0),
        metrics.get("ada_accessibility_score", 88.0),
        metrics.get("acoustic_nrc_rating", 0.65) * 100.0,
        metrics.get("ergonomic_efficiency_score", 92.0)
    ]
    
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(5.2, 5.2), subplot_kw=dict(polar=True), facecolor='#ffffff')
    ax.set_facecolor('#f8fafc')
    
    ax.plot(angles, values, color='#2563eb', linewidth=2.4, linestyle='solid')
    ax.fill(angles, values, color='#3b82f6', alpha=0.20)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9.5, fontweight='600', color='#1e293b')
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=7.5, color='#64748b')
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', color='#cbd5e1', alpha=0.8)
    ax.spines['polar'].set_color('#94a3b8')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def generate_metrics_bar_chart(metrics: Dict[str, float]) -> Image.Image:
    """Generates a clean horizontal bar chart for layout performance metrics."""
    categories = ['Circulation Flow', 'Lighting Index', 'Accessibility', 'Acoustic Rating', 'Ergonomics']
    scores = [
        metrics.get("circulation_flow_rate_pct", 88.5),
        metrics.get("lighting_score", 90.0),
        metrics.get("ada_accessibility_score", 88.0),
        metrics.get("acoustic_nrc_rating", 0.65) * 100.0,
        metrics.get("ergonomic_efficiency_score", 92.0)
    ]
    colors = ['#2563eb', '#0284c7', '#16a34a', '#d97706', '#7c3aed']

    fig, ax = plt.subplots(figsize=(6.2, 3.6), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')

    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, scores, align='center', color=colors, height=0.52, edgecolor='none')

    for bar, score in zip(bars, scores):
        ax.text(score + 1.5, bar.get_y() + bar.get_height()/2.0, f"{score:.1f}%",
                va='center', ha='left', fontsize=9, fontweight='bold', color='#1e293b')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=9, color='#334155', fontweight='600')
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xlabel('Performance Score (%)', fontsize=8.5, color='#64748b')
    ax.grid(axis='x', linestyle=':', color='#e2e8f0', alpha=0.8)
    
    for spine in ax.spines.values():
        spine.set_visible(False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def generate_spatial_clearance_heatmap(metrics: Dict[str, Any], detected_items: Optional[List[Dict[str, Any]]] = None) -> Image.Image:
    """Renders a clean 2D architectural blueprint layout diagram showing walking pathways and clearances."""
    fig, ax = plt.subplots(figsize=(7.5, 5.6), facecolor='#ffffff')
    ax.set_facecolor('#f8fafc')

    room_w, room_h = 800, 600
    ax.set_xlim(-40, room_w + 40)
    ax.set_ylim(-40, room_h + 40)

    # 1. Outer Wall Boundary
    wall_rect = patches.Rectangle((0, 0), room_w, room_h, linewidth=2.5, edgecolor='#334155', facecolor='#ffffff')
    ax.add_patch(wall_rect)

    # 2. Window (Top Wall - Natural Daylight)
    window = patches.Rectangle((250, room_h - 6), 300, 12, linewidth=1.8, edgecolor='#0284c7', facecolor='#38bdf8')
    ax.add_patch(window)
    ax.text(400, room_h + 15, "Window (Natural Light Ingress)", color='#0284c7', fontsize=8.5, ha='center', fontweight='bold')

    # Subtle daylight polygon
    daylight_polygon = patches.Polygon([[250, room_h], [550, room_h], [700, 200], [100, 200]], closed=True, color='#38bdf8', alpha=0.08)
    ax.add_patch(daylight_polygon)

    # 3. Entrance Door (Bottom Left)
    door = patches.Rectangle((60, -6), 92, 12, linewidth=1.8, edgecolor='#16a34a', facecolor='#4ade80')
    ax.add_patch(door)
    door_arc = patches.Arc((60, 0), 184, 184, angle=0, theta1=0, theta2=90, color='#16a34a', linestyle=':', linewidth=1.5)
    ax.add_patch(door_arc)
    ax.text(106, -22, "Doorway (92cm Entry)", color='#16a34a', fontsize=8, ha='center', fontweight='600')

    # 4. Furniture Footprints with Clean Modern Styling
    # Main Sofa
    sofa_buf = patches.Rectangle((160, 240), 380, 200, linewidth=1, linestyle='--', edgecolor='#16a34a', facecolor='#dcfce7', alpha=0.4)
    sofa = patches.Rectangle((180, 260), 340, 160, linewidth=1.5, edgecolor='#2563eb', facecolor='#dbeafe', alpha=0.95)
    ax.add_patch(sofa_buf)
    ax.add_patch(sofa)
    ax.text(350, 340, "Modular Sofa\n(340cm x 160cm)", color='#1e3a8a', fontsize=8, ha='center', va='center', fontweight='bold')

    # Coffee Table
    table_buf = patches.Rectangle((270, 120), 160, 90, linewidth=1, linestyle='--', edgecolor='#16a34a', facecolor='#dcfce7', alpha=0.4)
    table = patches.Rectangle((285, 135), 130, 60, linewidth=1.2, edgecolor='#64748b', facecolor='#f1f5f9', alpha=0.95)
    ax.add_patch(table_buf)
    ax.add_patch(table)
    ax.text(350, 165, "Coffee Table", color='#334155', fontsize=7.5, ha='center', va='center', fontweight='600')

    # Media Unit
    tv_unit = patches.Rectangle((room_w - 60, 180), 45, 240, linewidth=1.2, edgecolor='#d97706', facecolor='#fef3c7', alpha=0.95)
    ax.add_patch(tv_unit)
    ax.text(room_w - 38, 300, "Media Unit", color='#92400e', fontsize=7.5, ha='center', va='center', rotation=90, fontweight='600')

    # Armchair
    armchair = patches.Rectangle((620, 440), 100, 100, linewidth=1.2, edgecolor='#7c3aed', facecolor='#ede9fe', alpha=0.95)
    ax.add_patch(armchair)
    ax.text(670, 490, "Armchair", color='#5b21b6', fontsize=7.5, ha='center', va='center', fontweight='600')

    # Indoor Botanical Element
    plant = patches.Circle((90, 510), 45, linewidth=1.2, edgecolor='#16a34a', facecolor='#dcfce7', alpha=0.95)
    ax.add_patch(plant)
    ax.text(90, 510, "Plant", color='#166534', fontsize=7.5, ha='center', va='center', fontweight='600')

    # 5. Clearance Pathway Annotations
    ax.annotate("", xy=(220, 180), xytext=(110, 40),
                arrowprops=dict(arrowstyle="->", color='#0284c7', lw=1.5, linestyle='-.'))
    ax.text(120, 120, "Walkway: >90cm", color='#0284c7', fontsize=8, fontweight='bold')

    ax.plot([350, 350], [195, 260], color='#16a34a', linewidth=1.5, marker='|')
    ax.text(358, 227, "Clearance: 65cm", color='#16a34a', fontsize=7.5, va='center', fontweight='600')

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("2D Layout and Clearance Map (8m x 6m Grid)", fontsize=10.5, fontweight='bold', pad=14, color='#0f172a')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
