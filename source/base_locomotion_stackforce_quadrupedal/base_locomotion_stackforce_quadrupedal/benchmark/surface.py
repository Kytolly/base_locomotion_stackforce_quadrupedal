"""Analytic vertical support queries for the static benchmark collision primitives."""

import math

import torch


def track_surface_height(parameters, xy):
    """Return the highest surface at world XY, including the surrounding ground."""
    p = parameters
    x, y = xy.unbind(-1)
    height = torch.zeros_like(x)
    base = float(p.get("base_surface_height_m", 0.0))
    start = p["approach_m"]
    ramp = p["ramp_length_m"]
    rise = p["ramp_height_m"]
    bed_start = start + ramp
    bed_end = bed_start + p.get("washboard_length_m", p.get("plateau_length_m", 0.0))

    def slab(y0, y1, z0, z1):
        nonlocal height
        inside = (x.abs() <= p["width_m"] / 2) & (y >= y0) & (y <= y1)
        z = z0 + (y - y0) * (z1 - z0) / (y1 - y0)
        height = torch.where(inside, torch.maximum(height, z), height)

    slab(-0.3, start, base, base)
    slab(start, bed_start, base, base + rise)
    slab(bed_end, bed_end + ramp, base + rise, base)
    slab(bed_end + ramp, bed_end + ramp + p["release_m"], base, base)
    if p["kind"] == "plateau":
        slab(bed_start, bed_end, rise, rise)
        return height

    radius = p["cylinder_radius_m"]
    leading = p["row_stagger_offsets_m"].index(0.0)
    for row in range(2):
        for index in range(p["cylinders_per_row"]):
            dx, dy = p["cylinder_position_offsets_xy_m"][row][index]
            cx = p["lateral_row_centers_m"][row] + dx
            cy = bed_start + index * p["cylinder_spacing_m"] + p["row_stagger_offsets_m"][row] + dy
            cz = base + p["cylinder_height_offsets_m"][row][index] + radius
            entry = row == leading and index == 0
            exit_edge = row != leading and index == p["cylinders_per_row"] - 1
            if entry or exit_edge:
                cy += radius / 2 if entry else -radius / 2
                inside = ((x - cx).abs() <= p["cylinder_length_m"] / 2) & ((y - cy).abs() <= radius / 2)
                z = torch.full_like(height, cz + radius)
            else:
                angle = math.radians(p["cylinder_orientation_offsets_deg"][row][index])
                along = (x - cx) * math.cos(angle) + (y - cy) * math.sin(angle)
                across = -(x - cx) * math.sin(angle) + (y - cy) * math.cos(angle)
                inside = (along.abs() <= p["cylinder_length_m"] / 2) & (across.abs() <= radius)
                z = cz + torch.sqrt((radius * radius - across.square()).clamp_min(0.0))
            height = torch.where(inside, torch.maximum(height, z), height)
    return height
