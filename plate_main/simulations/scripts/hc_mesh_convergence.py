#!/usr/bin/env python3
"""
Mesh Convergence Study — 300mm CCCC Honeycomb Sandwich Panel
Includes midplane offset fix for face sheets + FreeTri mesh.
"""
import mph, numpy as np, time, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

N_EIGS = 10
PAPER_FEM = [1217.9, 2312.6, 2313.8, 3236.1, 3803.4, 3838.7, 4594.7, 4597.6, 5575.4, 5584.7]

# ============================================================
# GEOMETRY + PHYSICS (build once)
# ============================================================
client = mph.start(cores=2)
model = client.create('hc_conv_final')
m = model.java

LC = 0.003; HC = 0.008; PLATE = 0.3; HFACE = 0.001; TC = 0.0002
m.component().create("comp1", True)
m.param().set("lc", f"{LC}[m]")
m.param().set("hc", f"{HC}[m]")
m.param().set("L", f"{PLATE}[m]")
m.param().set("W", f"{PLATE}[m]")
m.param().set("h_face", f"{HFACE}[m]")
m.param().set("tc", f"{TC}[m]")

g = m.component("comp1").geom().create("geom1", 3)

g.create("wp1", "WorkPlane").set("quickplane", "xy")
wp = g.feature("wp1").geom()

hex1_pts = [(LC*np.cos(k*np.pi/3), LC*np.sin(k*np.pi/3)) for k in range(6)]
hex1_pts.append(hex1_pts[0])
wp.create("hex1", "Polygon").set("type", "open")
wp.feature("hex1").set("x", ",".join(f"{p[0]:.10e}" for p in hex1_pts))
wp.feature("hex1").set("y", ",".join(f"{p[1]:.10e}" for p in hex1_pts))

dx2, dy2 = 1.5*LC, np.sqrt(3)/2*LC
hex2_pts = [(LC*np.cos(k*np.pi/3)+dx2, LC*np.sin(k*np.pi/3)+dy2) for k in range(6)]
hex2_pts.append(hex2_pts[0])
wp.create("hex2", "Polygon").set("type", "open")
wp.feature("hex2").set("x", ",".join(f"{p[0]:.10e}" for p in hex2_pts))
wp.feature("hex2").set("y", ",".join(f"{p[1]:.10e}" for p in hex2_pts))

nx = int(np.ceil(PLATE/(3*LC)))+4
ny = int(np.ceil(PLATE/(np.sqrt(3)*LC)))+4
dx_a, dy_a = 3*LC, np.sqrt(3)*LC
for tag, sname in [("arr1","hex1"),("arr2","hex2")]:
    a = wp.create(tag, "Array")
    a.selection("input").set(sname)
    a.set("type", "rectangular")
    a.setIndex("fullsize", str(nx), 0)
    a.setIndex("fullsize", str(ny), 1)
    a.setIndex("displ", f"{dx_a:.10e}", 0)
    a.setIndex("displ", f"{dy_a:.10e}", 1)

wp.create("uni1", "Union").selection("input").set("arr1", "arr2")
r = wp.create("r1", "Rectangle")
r.set("size", ["L", "W"]); r.set("base", "center"); r.set("pos", ["L/2", "W/2"])
wp.create("int1", "Intersection").selection("input").set("uni1", "r1")

g.create("ext1", "Extrude").selection("input").set("wp1")
g.feature("ext1").setIndex("distance", "hc", 0)

# Bottom face sheet at z=0
g.create("wp_bot", "WorkPlane").set("quickplane", "xy")
rb = g.feature("wp_bot").geom().create("r1", "Rectangle")
rb.set("size", ["L", "W"]); rb.set("base", "center"); rb.set("pos", ["L/2", "W/2"])

# Top face sheet at z=hc
g.create("wp_top", "WorkPlane").set("quickplane", "xy").set("quickz", "hc")
rt = g.feature("wp_top").geom().create("r1", "Rectangle")
rt.set("size", ["L", "W"]); rt.set("base", "center"); rt.set("pos", ["L/2", "W/2"])

print("=" * 65)
print("MESH CONVERGENCE — 300mm CCCC Honeycomb (Offset Fix + FreeTri)")
print("=" * 65)
print("Building geometry...")
t0 = time.time()
g.run()
ne = g.getNEdges(); nb = g.getNBoundaries()
print(f"  B={nb}, E={ne} ({time.time()-t0:.1f}s)")

# Material
mat = m.component("comp1").material().create("mat1", "Common")
mat.propertyGroup("def").set("density", "2710")
mat.propertyGroup("def").set("youngsmodulus", "70e9")
mat.propertyGroup("def").set("poissonsratio", "0.33")
mat.selection().all()

# --- Selections ---
sel = m.component("comp1").selection()

# Face sheets (at z=0 and z=hc)
fb = sel.create("sel_bot", "Box")
fb.set("entitydim", "2"); fb.set("zmin", "-1e-5"); fb.set("zmax", "1e-5")
ft = sel.create("sel_top", "Box")
ft.set("entitydim", "2"); ft.set("zmin", "hc-1e-5"); ft.set("zmax", "hc+1e-5")

# Core walls: Box at mid-height (only vertical surfaces)
sc = sel.create("sel_core", "Box")
sc.set("entitydim", "2"); sc.set("zmin", "1e-4"); sc.set("zmax", "hc-1e-4")

# Outer edges
tol = "1e-5"
el = sel.create("edg_L", "Box"); el.set("entitydim", "1")
el.set("xmin", f"-{tol}"); el.set("xmax", f"{tol}")
el.set("condition", "inside")
er = sel.create("edg_R", "Box"); er.set("entitydim", "1")
er.set("xmin", f"L-{tol}"); er.set("xmax", f"L+{tol}")
er.set("condition", "inside")
eb = sel.create("edg_B", "Box"); eb.set("entitydim", "1")
eb.set("ymin", f"-{tol}"); eb.set("ymax", f"{tol}")
eb.set("condition", "inside")
et = sel.create("edg_T", "Box"); et.set("entitydim", "1")
et.set("ymin", f"W-{tol}"); et.set("ymax", f"W+{tol}")
et.set("condition", "inside")
eo = sel.create("sel_outer_edges", "Union"); eo.set("entitydim", "1")
eo.set("input", ["edg_L", "edg_R", "edg_B", "edg_T"])

# --- Shell Physics with Midplane Offset ---
phys = m.component("comp1").physics().create("sh", "Shell", "geom1")

# Bottom face sheet: offset DOWN by h_face/2
th_bot = phys.create("th_bot", "ThicknessOffset")
th_bot.selection().named("sel_bot")
th_bot.set("d", "h_face")
th_bot.set("OffsetDefinition", "RelativeDistance")
th_bot.set("z_offset_rel", "-1")

# Top face sheet: offset UP by h_face/2
th_top = phys.create("th_top", "ThicknessOffset")
th_top.selection().named("sel_top")
th_top.set("d", "h_face")
th_top.set("OffsetDefinition", "RelativeDistance")
th_top.set("z_offset_rel", "1")

# Core walls: no offset
th_c = phys.create("th_core", "ThicknessOffset")
th_c.selection().named("sel_core")
th_c.set("d", "tc")

# Fixed BC on outer edges
fix = phys.create("fix1", "Fixed", 1)
fix.selection().named("sel_outer_edges")

# --- Mesh (created once, hauto updated in loop) ---
mesh = m.component("comp1").mesh().create("mesh1")
ftri = mesh.create("ftri1", "FreeTri")
ftri.selection().all()
sz = ftri.create("size1", "Size")

# --- Study (created once) ---
study = m.study().create("std1")
study.create("eig", "Eigenfrequency")
study.feature("eig").set("neigs", "10")
study.feature("eig").set("shift", "1000[Hz]")

# ============================================================
# CONVERGENCE LOOP
# ============================================================
hauto_levels = [9, 8, 7, 6, 5, 4, 3, 2, 1]
num_elements = []
freq_results = {m: [] for m in range(1, N_EIGS+1)}
mesh_times = []
solve_times = []

save_dir = "D:/plate-main/simulations/comsol_results/mesh_convergence"
os.makedirs(save_dir, exist_ok=True)

print(f"\n{'='*65}")
print(f"{'hauto':>8} {'Elements':>12} {'Mesh(s)':>8} {'Solve(s)':>9} {'f1(Hz)':>8} {'f2(Hz)':>8} {'f3(Hz)':>8} {'f4(Hz)':>8} {'f5(Hz)':>8} {'f6(Hz)':>8} {'f7(Hz)':>8} {'f8(Hz)':>8} {'f9(Hz)':>8} {'f10(Hz)':>8}")
print(f"{'='*65}")

for h_val in hauto_levels:
    print(f"\n--- hauto = {h_val} ---")

    sz.set("hauto", str(h_val))

    # Rebuild mesh
    print("  Meshing...", end="", flush=True)
    t0 = time.time()
    try:
        mesh.run()
        mt = time.time() - t0
        mesh_times.append(mt)
        print(f" done ({mt:.1f}s)")
    except Exception as e:
        print(f" FAILED: {str(e)[:100]}")
        num_elements.append(0)
        for mode in range(1, N_EIGS+1): freq_results[mode].append(0)
        continue

    # Count elements
    try:
        stats = m.component("comp1").mesh("mesh1").stat()
        npt = int(str(stats.getNumElem()))
        num_elements.append(npt)
        print(f"  Elements: {npt}")
    except Exception as e:
        print(f"  Element count failed: {e}")
        num_elements.append(0)

    # Solve
    print("  Solving... (this may take several minutes)", flush=True)
    t0 = time.time()
    try:
        study.run()
        st = time.time() - t0
        solve_times.append(st)
        print(f"  Solved ({st:.1f}s)")
    except Exception as e:
        print(f"  FAILED: {str(e)[:100]}")
        for mode in range(1, N_EIGS+1): freq_results[mode].append(0)
        continue

    # Extract frequencies
    try:
        m.result().numerical().create("gev1", "EvalGlobal")
        m.result().numerical("gev1").set("expr", ["shell.freq"])
        m.result().numerical("gev1").set("descr", ["Eigenfrequency"])
        m.result().numerical("gev1").run()
        freqs = m.result().numerical("gev1").getReal()
        for mode in range(1, N_EIGS+1):
            freq_results[mode].append(freqs[0][mode-1])
        print(f"  f1={freqs[0][0]:.1f} f2={freqs[0][1]:.1f} f3={freqs[0][2]:.1f} f4={freqs[0][3]:.1f}")
        print(f"  f5={freqs[0][4]:.1f} f6={freqs[0][5]:.1f} f7={freqs[0][6]:.1f} f8={freqs[0][7]:.1f} f9={freqs[0][8]:.1f} f10={freqs[0][9]:.1f}")
        m.result().numerical().remove("gev1")
    except Exception as e:
        print(f"  Result extraction failed: {str(e)[:100]}")
        for mode in range(1, N_EIGS+1): freq_results[mode].append(0)

    mph_path = os.path.join(save_dir, f"hauto_{h_val}", "model.mph")
    os.makedirs(os.path.dirname(mph_path), exist_ok=True)
    model.save(mph_path)
    print(f"  Saved: {mph_path}")

    freqs_str = " | ".join([f"f{m}={freq_results[m][-1]:>8.1f}" for m in [1,2,3,4,5,6,7,8,9,10]])
    print(f"  hauto={h_val:>2} | {num_elements[-1]:>10} | {mesh_times[-1]:>7.1f}s | {solve_times[-1]:>8.1f}s | {freqs_str}")

# ============================================================
# SUMMARY + PLOT
# ============================================================
print(f"\n{'='*90}")
print("SUMMARY")
print(f"{'='*90}")
print(f"{'hauto':>8} {'Elements':>10}  ", end="")
for m in range(1, N_EIGS+1): print(f"{f'f{m}(Hz)':>9}", end=" ")
print()
for i, h in enumerate(hauto_levels):
    print(f"{h:>8} {num_elements[i]:>10}  ", end="")
    for m in range(1, N_EIGS+1):
        print(f"{freq_results[m][i]:>9.1f}", end=" ")
    print()

print(f"\nPaper FEM:  {PAPER_FEM} Hz (all 10 modes)")

# --- ERROR REPORT for each hauto ---
print(f"\n{'='*90}")
print("ERROR vs PAPER FEM (for each hauto)")
print(f"{'='*90}")

hauto_mean_errors = {}
hauto_max_errors = {}

for i, h in enumerate(hauto_levels):
    if num_elements[i] == 0 or freq_results[1][i] == 0:
        continue
    print(f"\n--- hauto={h} ({num_elements[i]} elements) ---")
    print(f"  {'Mode':>6} {'Our(Hz)':>10} {'Paper(Hz)':>10} {'Error(%)':>10}")
    print("  " + "-" * 40)
    errors = []
    for m in range(1, N_EIGS+1):
        f_ours = freq_results[m][i]
        f_paper = PAPER_FEM[m-1]
        err_pct = abs(f_ours - f_paper) / f_paper * 100
        errors.append(err_pct)
        print(f"  {m:>6}  {f_ours:>10.1f}  {f_paper:>10.1f}  {err_pct:>9.2f}%")
    print("  " + "-" * 40)
    mean_err = np.mean(errors)
    max_err = np.max(errors)
    hauto_mean_errors[h] = mean_err
    hauto_max_errors[h] = max_err
    print(f"  Mean error: {mean_err:.2f}%  |  Max error: {max_err:.2f}%")

# --- BEST hauto ---
best_h = min(hauto_mean_errors, key=hauto_mean_errors.get)
best_idx = hauto_levels.index(best_h)
print(f"\n{'='*90}")
print(f"BEST hauto = {best_h}  (mean error = {hauto_mean_errors[best_h]:.2f}%, max error = {hauto_max_errors[best_h]:.2f}%, {num_elements[best_idx]} elements)")
print(f"{'='*90}")

# --- RANKING ---
print(f"\n{'Rank':>6} {'hauto':>8} {'Elements':>10} {'Mean Err%':>10} {'Max Err%':>10}")
print("-" * 50)
for rank, h in enumerate(sorted(hauto_mean_errors, key=hauto_mean_errors.get), 1):
    idx = hauto_levels.index(h)
    print(f"  {rank:>4}  {h:>8}  {num_elements[idx]:>10}  {hauto_mean_errors[h]:>9.2f}%  {hauto_max_errors[h]:>9.2f}%")

# Save CSV
with open("D:/plate-main/mesh_convergence_final.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = ["hauto", "num_elements", "mesh_time_s", "solve_time_s"]
    header += [f"freq_mode{m}" for m in range(1, N_EIGS+1)]
    header += ["mean_error_%", "max_error_%"]
    writer.writerow(header)
    for i in range(len(hauto_levels)):
        row = [hauto_levels[i], num_elements[i], mesh_times[i], solve_times[i]]
        row += [freq_results[m][i] for m in range(1, N_EIGS+1)]
        h = hauto_levels[i]
        row += [hauto_mean_errors.get(h, 0), hauto_max_errors.get(h, 0)]
        writer.writerow(row)
print("Saved mesh_convergence_final.csv")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
valid = [i for i in range(len(num_elements)) if num_elements[i] > 0 and freq_results[1][i] > 0]

colors = ['b', 'r', 'g', 'm', 'c', 'orange', 'purple', 'brown', 'pink', 'olive']
markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p', '*', 'h']

# Left: freq vs hauto
ax = axes[0]
for m in range(1, N_EIGS+1):
    ax.plot([hauto_levels[i] for i in valid], [freq_results[m][i] for i in valid],
            color=colors[m-1], marker=markers[m-1], label=f'Mode {m}', markersize=6)
for m, f_paper in enumerate(PAPER_FEM, 1):
    ax.axhline(y=f_paper, color=colors[m-1], linestyle='--', alpha=0.4)
ax.set_xlabel('hauto Level (1=coarse, 9=fine)', fontsize=12)
ax.set_ylabel('Eigenfrequency (Hz)', fontsize=12)
ax.set_title('Mesh Convergence — Freq vs hauto', fontsize=13)
ax.legend(fontsize=8, loc='center right')
ax.grid(True, alpha=0.3)

# Right: freq vs element count
ax = axes[1]
for m in range(1, N_EIGS+1):
    ax.plot([num_elements[i] for i in valid], [freq_results[m][i] for i in valid],
            color=colors[m-1], marker=markers[m-1], label=f'Mode {m}', markersize=6)

for m, f_paper in enumerate(PAPER_FEM, 1):
    ax.axhline(y=f_paper, color=colors[m-1], linestyle='--', alpha=0.4)
ax.set_xlabel('Number of Elements', fontsize=12)
ax.set_ylabel('Eigenfrequency (Hz)', fontsize=12)
ax.set_title('Mesh Convergence — Freq vs Elements', fontsize=13)
ax.legend(fontsize=8, loc='center right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("D:/plate-main/mesh_convergence_final.png", dpi=150)
print("Saved mesh_convergence_final.png")

# client.remove(model)
# client.clear()
print("\nDone!")
