"""Opt-in licensed COMSOL smoke: python test_hf_smoke.py. No dataset writes."""
import numpy as np
def main():
    import mph
    import hc_HighFidelity_LHS as hf
    from p1_geometry import physical_parameters

    p = physical_parameters(dict(alpha=.8, beta=.5, theta_c=40,
                                 eta1=1.3, eta2=.0666667))
    p['L'] = .012
    print('Starting tiny COMSOL integration smoke', flush=True)
    client = None
    model = None
    try:
        client = mph.start(cores=2)
        print('COMSOL connected', flush=True)
        model = hf.build_model(client, p, mesh_size=9, n_eigs=3,
                               candidate_eigs=6, eigen_shift_hz=1000)
        print('Tiny model built; solving', flush=True)
        data = hf.solve_and_extract(model, p, n_eigs=3, grid_res=5)
        assert data['w'].shape == (3, 5, 5)
        assert np.all(np.isfinite(data['frequencies']))
        assert np.all(data['transverse_fraction'] >= hf.MIN_TRANSVERSE_FRACTION)
        assert np.allclose(data['w'][:, [0, -1], :], 0)
        assert np.allclose(data['w'][:, :, [0, -1]], 0)
        print('PASS', data['frequencies'], data['w'].shape, flush=True)
    finally:
        if model is not None:
            try:
                client.remove(model)
            except Exception:
                pass
        if client is not None:
            try:
                client.clear()
            except Exception:
                pass
            try:
                client.disconnect()
            except Exception:
                pass


if __name__ == '__main__':
    main()
