#!/usr/bin/env python3
"""Versioned P1 LF modal bundles from existing samples; never resample or renumber."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import plate
import honeycomb

ROOT = Path(__file__).resolve().parent
PARAMETERS = ('alpha','beta','theta_c','eta1','eta2')
L1 = L2 = 0.3
H_TOTAL = 0.01
E_FACE, RHO_FACE, NU_FACE = 70e9, 2710., 0.33
G_FACE = E_FACE/(2*(1+NU_FACE))
K_SPRING = 1e12
FLEXURAL_FRACTION_MIN = 0.5


def validate_parameters(p):
    if set(p) != set(PARAMETERS) or not all(np.isfinite(v) for v in p.values()):
        raise ValueError('Each sample needs five finite P1 parameters')
    if not 0 < p['alpha'] < 1 or p['beta'] <= 0:
        raise ValueError('Require 0 < alpha < 1 and beta > 0')
    if not 0 <= p['theta_c'] < 90 or p['eta1'] <= 0 or p['eta2'] <= 0:
        raise ValueError('Require angle in [0,90) degrees and positive eta1,eta2')


def read_samples(path):
    samples, seen = [], set()
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not set(PARAMETERS).issubset(reader.fieldnames or ()):
            raise ValueError('Sample CSV must contain '+', '.join(PARAMETERS))
        id_column = next((k for k in ('run_id','run') if k in reader.fieldnames), None)
        for row_index, row in enumerate(reader, 1):
            run_id = int(row[id_column]) if id_column else row_index
            if run_id < 1 or run_id in seen:
                raise ValueError(f'Invalid or duplicate run ID: {run_id}')
            p = {key: float(row[key]) for key in PARAMETERS}
            validate_parameters(p)
            samples.append((run_id,p))
            seen.add(run_id)
    if not samples:
        raise ValueError('Empty sample table')
    return samples


def code_hash():
    digest = hashlib.sha256()
    paths = [Path(__file__), ROOT/'honeycomb.py', *sorted((ROOT/'plate').glob('*.py'))]
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_plate(p, order):
    validate_parameters(p)
    if order < 3:
        raise ValueError('Ritz order must be at least 3')
    hc = H_TOTAL*p['alpha']
    # beta is top-face / bottom-face thickness; layers are bottom to top.
    h_bottom = (H_TOTAL-hc)/(1+p['beta'])
    z = np.array([0.,h_bottom,h_bottom+hc,H_TOTAL])-H_TOTAL/2
    face = plate.Material(E_FACE,E_FACE,G_FACE,G_FACE,G_FACE,NU_FACE,RHO_FACE)
    core = plate.Material(*honeycomb.material_property(E_FACE,G_FACE,RHO_FACE,
        p['eta2']*3e-3,3e-3,p['eta1']*3e-3,np.deg2rad(p['theta_c'])))
    profile = plate.Profile([face,core,face],[0,0,0],z)
    model = plate.Plate(L1,L2,order,order,profile)
    model.set_basis(plate.legendre_basis(order,interval=[0,L1]),
                    plate.legendre_basis(order,interval=[0,L2]))
    for axis,limit in (('x',L1),('y',L2)):
        for coordinate in (0.,limit):
            model.add_spring(K_SPRING,plate.BC_C,**{axis:coordinate})
    return model


def _real_coefficients(frame):
    coeff = np.asarray(frame._u)
    if np.max(np.abs(coeff.imag)) > 1e-9*max(np.max(np.abs(coeff.real)),1e-300):
        raise ValueError('Structural eigenmode has material imaginary component')
    return coeff.real.reshape(5,frame._plate._M,frame._plate._N)


def _transverse_fraction(model,coeff):
    # Translational kinetic energy; common through-thickness I0 cancels.
    values = np.einsum('xi,bij,yj->bxy',model._basis_x_values[0].T,
                      coeff[:3],model._basis_y_values[0].T,optimize=True)
    weights = model._quad_x.w[:,None]*model._quad_y.w[None,:]
    energies = np.einsum('bxy,xy->b',values**2,weights)
    return float(energies[2]/energies.sum()) if energies.sum()>0 else 0.


def compute_bundle(parameters,modes=16,order=15,grid=80):
    if modes<1 or grid<3:
        raise ValueError('Require positive modes and grid >= 3')
    model = build_plate(parameters,order)
    model.assemble()
    ndof = 5*order*order
    count = min(max(2*modes,modes+12),ndof-2)
    max_count = min(max(4*modes,modes+24),ndof-2)
    while True:
        result = plate.ModalSolver(model).solve(count)
        accepted = []
        for index,frame in enumerate(result,1):
            frequency = float(frame.frequency)
            if not np.isfinite(frequency) or frequency <= 0:
                continue
            coeff = _real_coefficients(frame)
            fraction = _transverse_fraction(model,coeff)
            if fraction >= FLEXURAL_FRACTION_MIN:
                accepted.append((frequency,coeff,index,fraction))
            if len(accepted)==modes:
                break
        if len(accepted)==modes:
            break
        if count==max_count:
            raise ValueError(f'Only {len(accepted)} flexural modes among {count}; requested {modes}')
        count = max_count
    x,y = np.linspace(0,L1,grid),np.linspace(0,L2,grid)
    bx = np.array([model._basis_x[0](xi) for xi in x])
    by = np.array([model._basis_y[0](yi) for yi in y])
    w = np.array([by@coeff[2].T@bx.T for _,coeff,_,_ in accepted])
    peak = np.max(np.abs(w),axis=(1,2))
    if not np.isfinite(w).all() or np.any(peak<=0):
        raise ValueError('Selected transverse mode is zero or nonfinite')
    w /= peak[:,None,None]
    return dict(x=x,y=y,w=w,frequencies=np.array([v[0] for v in accepted]),
                eigen_mode_indices=np.array([v[2] for v in accepted],dtype=int),
                transverse_fraction=np.array([v[3] for v in accepted]))


def validate_bundle(arrays,modes,grid):
    for name in ('x','y','w','frequencies','transverse_fraction','eigen_mode_indices'):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f'Nonfinite {name}')
    if arrays['x'].shape != (grid,) or arrays['y'].shape != (grid,):
        raise ValueError('Wrong coordinate axis shape')
    if arrays['w'].shape != (modes,grid,grid) or arrays['frequencies'].shape != (modes,):
        raise ValueError('Incomplete modal bundle')
    if any(not np.all(np.diff(arrays[axis])>0) for axis in ('x','y')):
        raise ValueError('Coordinate axes must increase strictly')
    if np.any(arrays['frequencies']<=0) or np.any(np.diff(arrays['frequencies'])<0):
        raise ValueError('Frequencies must be positive and sorted')
    if np.any(np.max(np.abs(arrays['w']),axis=(1,2))<=0):
        raise ValueError('Zero transverse mode')
    if any(arrays[key].shape != (modes,) for key in ('eigen_mode_indices','transverse_fraction')):
        raise ValueError('Missing mode selection provenance')
    if np.any(arrays['transverse_fraction']<FLEXURAL_FRACTION_MIN):
        raise ValueError('Nonflexural mode in bundle')


def atomic_save(path,arrays,metadata):
    validate_bundle(arrays,metadata['modes'],metadata['grid'])
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.tmp',delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle,**arrays,metadata=np.array(json.dumps(metadata,sort_keys=True)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def reusable(path,metadata):
    try:
        with np.load(path,allow_pickle=False) as data:
            if json.loads(str(data['metadata'].item())) != metadata:
                return False
            validate_bundle(data,metadata['modes'],metadata['grid'])
        return True
    except (OSError,ValueError,KeyError,TypeError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,default=ROOT/'lhs_samples_v2.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'p1_data')
    parser.add_argument('--runs',help='Comma-separated original run IDs')
    parser.add_argument('--modes',type=int,default=16)
    parser.add_argument('--order',type=int,default=15)
    parser.add_argument('--grid',type=int,default=80)
    args = parser.parse_args(argv)
    if args.modes<1 or args.order<3 or args.grid<3:
        parser.error('Require modes >= 1, order >= 3, grid >= 3')
    samples = read_samples(args.samples)
    if args.runs:
        requested = {int(v) for v in args.runs.split(',')}
        missing = requested-{run for run,_ in samples}
        if missing:
            parser.error(f'Unknown run IDs: {sorted(missing)}')
        samples = [(run,p) for run,p in samples if run in requested]
    input_digest = hashlib.sha256(args.samples.read_bytes()).hexdigest()
    source_digest = code_hash()
    failed = []
    for run_id,parameters in samples:
        metadata = dict(schema_version=2,fidelity='lf',model_version='p1-v2',run_id=run_id,
            parameters=parameters,surface='midplane',extraction='ritz',order=args.order,
            grid=args.grid,modes=args.modes,input_hash=input_digest,code_hash=source_digest,
            mode_selection='transverse_translational_energy_fraction>=0.5',
            normalization='max_abs',boundary='CCCC_penalty_1e12',dimensions_m=[L1,L2,H_TOTAL])
        path = args.output/'lf'/f'run_{run_id:04d}.npz'
        if reusable(path,metadata):
            print(f'Run {run_id}: complete matching bundle, skipped',flush=True)
            continue
        started = time.monotonic()
        try:
            arrays = compute_bundle(parameters,args.modes,args.order,args.grid)
            atomic_save(path,arrays,metadata)
            print(f"Run {run_id}: {arrays['frequencies'][0]:.3f} Hz; saved {path} ({time.monotonic()-started:.1f}s)",flush=True)
        except Exception as error:
            failed.append(run_id)
            print(f'Run {run_id}: FAILED: {error}',flush=True)
    print(f'Processed {len(samples)} selected runs; failed: {failed}',flush=True)
    return 1 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())
