#!/usr/bin/env python3
"""
High-Fidelity LHS Automation — Shell Honeycomb Sandwich Panel
=============================================================
Loops through LHS samples (dimensionless), converts to physical parameters,
builds parametric shell honeycomb geometry, solves eigenfrequency study,
extracts frequencies + top face sheet mode shapes via fromdataset export.

Conversion from FSDT dimensionless params to physical:
  l1 = 3mm (fixed), L = 300mm (fixed), H_TOTAL = 10mm (fixed)
  lc  = l1 = 3mm
  tc  = eta2 * l1
  h_face = average of h1 and h3
  hc  = H_TOTAL * alpha
  theta_c = theta_c (same)
"""

import mph
import numpy as np
import time
import csv
import os

# ============================================================
# CONFIGURATION
# ============================================================
N_EIGS = 10           # Eigenfrequencies to solve
EIG_SHIFT = 1000      # Hz
GRID_RES = 80         # 80x80 grid for mode shape extraction
START_FROM = 1        # Resume from this run number (1 = start from beginning)

# Fixed reference values (must match FSDT solver)
L_FIXED = 0.3         # Plate side length (m)
H_TOTAL = 0.01        # Total plate thickness (m)
L1_FIXED = 0.003      # Cell edge length 1 (m)

LHS_FILE = "D:/plate-main/lhs_samples.csv"
RESULTS_CSV = "D:/plate-main/lhs_results_master.csv"
MODESHAPE_DIR = "D:/plate-main/Simulation_ModeShapes/"


def fsdt_to_physical(alpha, beta, theta_c, eta1, eta2):
    """Convert FSDT dimensionless parameters to physical parameters for COMSOL."""
    # Core wall thickness
    tc = eta2 * L1_FIXED

    # Cell edge lengths
    l1 = L1_FIXED
    l2 = l1 * eta1

    # lc = circumradius = edge length for regular hexagon
    lc = l1

    # Layer thicknesses
    h2 = H_TOTAL * alpha                       # Core height
    h1 = (H_TOTAL - h2) / (1 + beta) * beta   # Top face
    h3 = H_TOTAL - h1 - h2                     # Bottom face

    # Face sheet thickness (average for symmetric approximation)
    h_face = (h1 + h3) / 2

    return {
        'lc': lc,
        'tc': tc,
        'h_face': h_face,
        'hc': h2,
        'theta_c': theta_c,
        'L': L_FIXED,
    }


def build_model(client, params):
    lc = params['lc']
    tc = params['tc']
    h_face = params['h_face']
    hc = params['hc']
    theta_c_deg = params['theta_c']
    L = params['L']
    W = L

    model = client.create('lhs_run')
    m = model.java

    m.param().set("lc", f"{lc}[m]")
    m.param().set("tc", f"{tc}[m]")
    m.param().set("h_face", f"{h_face}[m]")
    m.param().set("hc", f"{hc}[m]")
    m.param().set("theta_c", f"{theta_c_deg}[deg]")
    m.param().set("L", f"{L}[m]")
    m.param().set("W", f"{W}[m]")

    m.component().create("comp1", True)
    g = m.component("comp1").geom().create("geom1", 3)

    # Honeycomb on WorkPlane
    g.create("wp1", "WorkPlane").set("quickplane", "xy")
    wp = g.feature("wp1").geom()

    hex1_pts = [(lc * np.cos(k * np.pi / 3), lc * np.sin(k * np.pi / 3)) for k in range(6)]
    hex1_pts.append(hex1_pts[0])
    wp.create("hex1", "Polygon").set("type", "open")
    wp.feature("hex1").set("x", ",".join(f"{p[0]:.10e}" for p in hex1_pts))
    wp.feature("hex1").set("y", ",".join(f"{p[1]:.10e}" for p in hex1_pts))

    dx2, dy2 = 1.5 * lc, np.sqrt(3) / 2 * lc
    hex2_pts = [(lc * np.cos(k * np.pi / 3) + dx2, lc * np.sin(k * np.pi / 3) + dy2) for k in range(6)]
    hex2_pts.append(hex2_pts[0])
    wp.create("hex2", "Polygon").set("type", "open")
    wp.feature("hex2").set("x", ",".join(f"{p[0]:.10e}" for p in hex2_pts))
    wp.feature("hex2").set("y", ",".join(f"{p[1]:.10e}" for p in hex2_pts))

    nx = int(np.ceil(L / (3 * lc))) + 4
    ny = int(np.ceil(L / (np.sqrt(3) * lc))) + 4
    dx_a, dy_a = 3 * lc, np.sqrt(3) * lc
    for tag, sname in [("arr1", "hex1"), ("arr2", "hex2")]:
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

    g.run()
    nb = g.getNBoundaries()

    # Material
    mat = m.component("comp1").material().create("mat1", "Common")
    mat.propertyGroup("def").set("density", "2710")
    mat.propertyGroup("def").set("youngsmodulus", "70e9")
    mat.propertyGroup("def").set("poissonsratio", "0.33")
    mat.selection().all()

    # Selections
    sel = m.component("comp1").selection()

    fb = sel.create("sel_bot", "Box")
    fb.set("entitydim", "2"); fb.set("zmin", "-1e-5"); fb.set("zmax", "1e-5")

    ft = sel.create("sel_top", "Box")
    ft.set("entitydim", "2"); ft.set("zmin", "hc-1e-5"); ft.set("zmax", "hc+1e-5")

    sc = sel.create("sel_core", "Box")
    sc.set("entitydim", "2"); sc.set("zmin", "1e-4"); sc.set("zmax", "hc-1e-4")

    tol = "1e-5"
    el = sel.create("edg_L", "Box"); el.set("entitydim", "1")
    el.set("xmin", f"-{tol}"); el.set("xmax", f"{tol}"); el.set("condition", "inside")
    er = sel.create("edg_R", "Box"); er.set("entitydim", "1")
    er.set("xmin", f"L-{tol}"); er.set("xmax", f"L+{tol}"); er.set("condition", "inside")
    eb = sel.create("edg_B", "Box"); eb.set("entitydim", "1")
    eb.set("ymin", f"-{tol}"); eb.set("ymax", f"{tol}"); eb.set("condition", "inside")
    et = sel.create("edg_T", "Box"); et.set("entitydim", "1")
    et.set("ymin", f"W-{tol}"); et.set("ymax", f"W+{tol}"); et.set("condition", "inside")
    eo = sel.create("sel_outer_edges", "Union"); eo.set("entitydim", "1")
    eo.set("input", ["edg_L", "edg_R", "edg_B", "edg_T"])

    # Shell Physics
    phys = m.component("comp1").physics().create("sh", "Shell", "geom1")

    th_bot = phys.create("th_bot", "ThicknessOffset")
    th_bot.selection().named("sel_bot")
    th_bot.set("d", "h_face")
    th_bot.set("OffsetDefinition", "RelativeDistance")
    th_bot.set("z_offset_rel", "-1")

    th_top = phys.create("th_top", "ThicknessOffset")
    th_top.selection().named("sel_top")
    th_top.set("d", "h_face")
    th_top.set("OffsetDefinition", "RelativeDistance")
    th_top.set("z_offset_rel", "1")

    th_c = phys.create("th_core", "ThicknessOffset")
    th_c.selection().named("sel_core")
    th_c.set("d", "tc")

    fix = phys.create("fix1", "Fixed", 1)
    fix.selection().named("sel_outer_edges")

    # Mesh
    mesh = m.component("comp1").mesh().create("mesh1")
    ftri = mesh.create("ftri1", "FreeTri")
    ftri.selection().all()
    sz = ftri.create("size1", "Size")
    sz.set("hauto", "4")
    mesh.run()

    # Study
    study = m.study().create("std1")
    study.create("eig", "Eigenfrequency")
    study.feature("eig").set("neigs", str(N_EIGS))
    study.feature("eig").set("shift", f"{EIG_SHIFT}[Hz]")

    return model, m, nb


def solve_and_extract(model, m, run_idx):
    """Solve, extract frequencies, then export mode shapes on 80x80 grid."""
    t0 = time.time()
    model.java.study("std1").run()
    solve_time = time.time() - t0

    res = m.result()

    # --- Extract eigenfrequencies ---
    res.numerical().create("gev1", "EvalGlobal")
    res.numerical("gev1").set("expr", ["shell.freq"])
    res.numerical("gev1").set("descr", ["Eigenfrequency"])
    res.numerical("gev1").run()
    freqs = res.numerical("gev1").getReal()
    res.numerical().remove("gev1")

    f_modes = [freqs[0][i] for i in range(N_EIGS)]

    # --- Fixed 80x80 grid for all runs ---
    os.makedirs(MODESHAPE_DIR, exist_ok=True)
    G = GRID_RES
    xv = np.linspace(0, L_FIXED, G)
    yv = np.linspace(0, L_FIXED, G)
    Xg, Yg = np.meshgrid(xv, yv)
    export_tag = f"data_exp{run_idx}"

    for mode_idx in range(1, N_EIGS + 1):
        raw_path = f"{MODESHAPE_DIR}raw_run{run_idx}_mode{mode_idx}.csv"
        int_path = f"{MODESHAPE_DIR}mode_shape_run{run_idx}_mode{mode_idx}.csv"
        print(f"    Mode {mode_idx}...", end="", flush=True)

        try:
            # Step 1: Export raw mesh nodes from COMSOL
            raw_temp = raw_path + ".tmp"
            exp = res.export().create(export_tag, "Data")
            exp.set("data", "dset1")
            exp.set("location", "fromdataset")
            exp.set("solnum", str(mode_idx))
            exp.set("expr", ["u", "v", "w", "x", "y", "z"])
            exp.set("filename", raw_temp)
            exp.run()
            res.export().remove(export_tag)
            if os.path.exists(raw_path):
                os.remove(raw_path)
            os.rename(raw_temp, raw_path)

            # Step 2: Read raw data and interpolate onto 80x80 grid
            raw = np.loadtxt(raw_path, skiprows=9)
            os.remove(raw_path)

            if raw.size == 0 or raw.ndim != 2:
                print(f"    Mode {mode_idx}: no data, skipping")
                continue

            # Columns: X, Y, Z, u, v, w, x, y, z
            # Use columns 0,1 (mesh coords) for positions, 3,4,5 for displacements
            pts = raw[:, :2]  # X, Y
            vals_u = raw[:, 3]
            vals_v = raw[:, 4]
            vals_w = raw[:, 5]

            # Interpolate onto 80x80 grid
            from scipy.interpolate import griddata
            Ui = griddata(pts, vals_u, (Xg, Yg), method="linear", fill_value=0)
            Vi = griddata(pts, vals_v, (Xg, Yg), method="linear", fill_value=0)
            Wi = griddata(pts, vals_w, (Xg, Yg), method="linear", fill_value=0)

            # Step 3: Save interpolated grid with columns: x, y, u, v, w
            nans = np.isnan(Ui) | np.isnan(Vi) | np.isnan(Wi)
            Ui[nans] = 0
            Vi[nans] = 0
            Wi[nans] = 0

            out = np.column_stack([Xg.ravel(), Yg.ravel(), Ui.ravel(), Vi.ravel(), Wi.ravel()])
            header = f"Run {run_idx} Mode {mode_idx} @ {f_modes[mode_idx-1]:.1f} Hz\nx,y,u,v,w\n{G}x{G} grid"
            int_temp = int_path + ".tmp"
            np.savetxt(int_temp, out, delimiter=",", header=header, comments="")
            if os.path.exists(int_path):
                os.remove(int_path)
            os.rename(int_temp, int_path)
            print(f" OK", end="", flush=True)

        except Exception as e:
            print(f" FAILED: {e}", end="", flush=True)
    print()

    return f_modes, solve_time


if __name__ == "__main__":
    print("=" * 70)
    print("HIGH-FIDELITY LHS AUTOMATION — Shell Honeycomb Sandwich Panel")
    print("=" * 70)

    # Load original LHS samples (dimensionless)
    samples_raw = []
    with open(LHS_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples_raw.append({k: float(v) for k, v in row.items()})

    print(f"Loaded {len(samples_raw)} LHS samples from {LHS_FILE}")
    print(f"Converting dimensionless -> physical params (L={L_FIXED*1000:.0f}mm, H={H_TOTAL*1000:.0f}mm)")
    print(f"Config: hauto=4, modes={N_EIGS}, grid={GRID_RES}x{GRID_RES}")

    # Convert to physical parameters
    samples = []
    for raw in samples_raw:
        phys = fsdt_to_physical(raw['alpha'], raw['beta'], raw['theta_c'],
                                raw['eta1'], raw['eta2'])
        phys['alpha'] = raw['alpha']
        phys['beta'] = raw['beta']
        phys['eta1'] = raw['eta1']
        phys['eta2'] = raw['eta2']
        samples.append(phys)

    all_results = []

    # Always try to load existing results for resume
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_results.append({k: float(v) if k != 'run' else int(v) for k, v in row.items()})
        print(f"Loaded {len(all_results)} existing results")

    fieldnames = ['run', 'alpha', 'beta', 'theta_c', 'eta1', 'eta2',
                  'lc', 'tc', 'h_face', 'hc',
                  'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10',
                  'solve_time_s', 'total_time_s']

    # Determine which runs already exist
    completed_runs = set(int(r['run']) for r in all_results)

    client = mph.start(cores=2)

    for idx, params in enumerate(samples):
        run_num = idx + 1
        if run_num in completed_runs:
            continue
        print(f"\n{'='*70}")
        print(f"  RUN {run_num}/{len(samples)}: lc={params['lc']:.4f} tc={params['tc']:.5f} "
              f"hf={params['h_face']:.4f} hc={params['hc']:.4f} "
              f"th={params['theta_c']:.1f} L={params['L']:.4f}")
        print(f"{'='*70}")

        t_start = time.time()
        print("  Building...", end="", flush=True)
        t_start = time.time()
        model, m, nb = build_model(client, params)
        print(f" B={nb}", end="", flush=True)

        print(" Solving...", end="", flush=True)
        f_modes, solve_time = solve_and_extract(model, m, run_num)
        total_time = time.time() - t_start
        print(f" done (solve={solve_time:.1f}s, total={total_time:.1f}s)")

        print(f"  f1={f_modes[0]:.1f} f2={f_modes[1]:.1f} f3={f_modes[2]:.1f} f4={f_modes[3]:.1f} "
              f"f5={f_modes[4]:.1f} f6={f_modes[5]:.1f} f7={f_modes[6]:.1f} f8={f_modes[7]:.1f} "
              f"f9={f_modes[8]:.1f} f10={f_modes[9]:.1f}")

        result = {
            'run': run_num,
            'alpha': params['alpha'], 'beta': params['beta'],
            'theta_c': params['theta_c'], 'eta1': params['eta1'], 'eta2': params['eta2'],
            'lc': params['lc'], 'tc': params['tc'], 'h_face': params['h_face'],
            'hc': params['hc'],
            'f1': f_modes[0], 'f2': f_modes[1], 'f3': f_modes[2], 'f4': f_modes[3],
            'f5': f_modes[4], 'f6': f_modes[5], 'f7': f_modes[6], 'f8': f_modes[7],
            'f9': f_modes[8], 'f10': f_modes[9],
            'solve_time_s': solve_time, 'total_time_s': total_time,
        }
        all_results.append(result)

        # Save model
        mph_path = f"D:/plate-main/lhs_run_{run_num}.mph"
        model.save(mph_path)
        print(f"  Saved: {mph_path}")
        client.remove(model)

        # Incremental CSV save
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Run':>4s} | {'f1(Hz)':>8s} {'f2(Hz)':>8s} {'f3(Hz)':>8s} {'f4(Hz)':>8s} {'f5(Hz)':>8s} {'f6(Hz)':>8s} {'f7(Hz)':>8s} {'f8(Hz)':>8s} {'f9(Hz)':>8s} {'f10(Hz)':>8s} | {'Time':>6s}")
    print("-" * 110)
    for r in all_results:
        print(f"  {r['run']:>2d}  | {r['f1']:>8.1f} {r['f2']:>8.1f} {r['f3']:>8.1f} {r['f4']:>8.1f} "
              f"{r['f5']:>8.1f} {r['f6']:>8.1f} {r['f7']:>8.1f} {r['f8']:>8.1f} {r['f9']:>8.1f} {r['f10']:>8.1f} | {r['solve_time_s']:>5.1f}s")

    client.clear()
    print(f"\nDone! Results: {RESULTS_CSV}")
    print(f"Mode shapes: {MODESHAPE_DIR}")
