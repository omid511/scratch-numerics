"""Fixed plate basis and continuous velocity-dependent matrix propagation."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.linalg import eigh,solve,eig
from scipy.integrate import solve_ivp
from mechanics.p4_margin_estimation.design_sampler import make_solver_from_design
from mechanics.p4_margin_estimation.transient import compute_eigendecomposition
from mechanics.p4_margin_estimation.operational import propagate_state
from mechanics.eigenanalysis import _transverse_participation,TRANSIENT_FILTER

PROFILES={
 'slow_cross':([0.,5.,35.,45.],[.9,.9,1.03,1.03]),
 'fast_cross':([0.,5.,10.,20.],[.9,.9,1.03,1.03]),
 'safe_dwell':([0.,45.],[.9,.9]),
 'near_dwell':([0.,45.],[.995,.995]),
 'approach_retreat':([0.,5.,20.,25.,40.,45.],[.9,.9,.995,.995,.9,.9])}
FORCING=('initial_only','intermittent_physical_pulses','weak_critical_initial')
NOISE=('clean','iid1','correlated1','iid5','correlated5','colored_common','gain_drift','impulses','dropout')


def ratio_at(profile,time):
    knots,values=PROFILES[profile]
    return np.interp(time,knots,values)


def crossing_time(profile):
    knots,values=PROFILES[profile]
    for a,b,va,vb in zip(knots[:-1],knots[1:],values[:-1],values[1:]):
        if va<1<=vb: return float(a+(1-va)/(vb-va)*(b-a))
    return None


def physical_pulse(time):
    # One-newton transverse point load;20ms half-sine every2s from1s.
    phase=(time-1.)%2.
    return float(np.sin(np.pi*phase/.02)) if time>=1. and phase<.02 else 0.


class PlateDynamics:
    def __init__(self,design,ucrit,sensor_yx,count=32):
        self.design=design; self.ucrit=float(ucrit); self.solver=make_solver_from_design(design)
        self.sensor_yx=np.asarray(sensor_yx,dtype=int)
        velocity=.9*self.ucrit
        self.M,K,C=self.solver.assemble_aeroelastic_system(velocity,design.rho_air,design.c_air,design.zeta)
        Ka,Ca=self.solver.assemble_aerodynamic(velocity,rho=design.rho_air,c_sound=design.c_air)
        self.Kbase=K-Ka; self.Cbase=C-Ca
        mach=velocity/design.c_air; dynamic=design.rho_air*velocity**2/np.sqrt(mach**2-1)
        damping=dynamic*(mach**2-2)/(mach**2-1)/velocity
        self.Ktemplate=Ka/dynamic; self.Ctemplate=Ca/damping
        eigenvalues,self.basis=eigh(self.Kbase,self.M,check_finite=True)
        if np.any(eigenvalues<=0): raise ValueError('structural basis has nonpositive frequency')
        self.omega=np.sqrt(eigenvalues); self.count=min(count,len(self.omega))
        mn=self.solver._MN_eff
        self.sensor=np.zeros((len(sensor_yx),len(eigenvalues)))
        for i,(iy,ix) in enumerate(sensor_yx):
            self.sensor[i,2*mn:3*mn]=np.outer(self.solver._vx_grid[:,ix],self.solver._vy_grid[:,iy]).ravel()
        # Physical transverse load, no solver pole identities in the forcing.
        load=np.zeros(len(eigenvalues)); iy=ix=16
        load[2*mn:3*mn]=np.outer(self.solver._vx_grid[:,ix],self.solver._vy_grid[:,iy]).ravel()
        self.load=load; self.reduced=self.matrices(self.count); self.full=self.matrices(len(eigenvalues))
        self.initial_modes=compute_eigendecomposition(self.solver,velocity,n_modes=8,sensor_xy=np.asarray(sensor_yx),rho=design.rho_air,c_sound=design.c_air,zeta=design.zeta)

    def matrices(self,count):
        phi=self.basis[:,:count]; omega=self.omega[:count]; mass=phi.T@self.M@phi
        def project(matrix): return solve(mass,phi.T@matrix@phi,assume_a='pos')
        base=np.zeros((2*count,2*count)); base[:count,count:]=np.diag(omega)
        base[count:,:count]=-project(self.Kbase)/omega[None,:]; base[count:,count:]=-project(self.Cbase)
        stiffness=np.zeros_like(base); stiffness[count:,:count]=-project(self.Ktemplate)/omega[None,:]
        damping=np.zeros_like(base); damping[count:,count:]=-project(self.Ctemplate)
        projection=np.column_stack((self.sensor@phi/omega[None,:],np.zeros((len(self.sensor),count))))
        force=np.r_[np.zeros(count),solve(mass,phi.T@self.load,assume_a='pos')]
        return dict(base=base,stiffness=stiffness,damping=damping,projection=projection,force=force,mass=mass)

    def matrix(self,ratio,count=None):
        matrices=self.reduced if count is None or count==self.count else self.full
        velocity=float(ratio)*self.ucrit; mach=velocity/self.design.c_air
        if mach<=1: raise ValueError('trajectory left supersonic piston-theory domain')
        dynamic=self.design.rho_air*velocity**2/np.sqrt(mach**2-1)
        damping=dynamic*(mach**2-2)/(mach**2-1)/velocity
        return matrices['base']+dynamic*matrices['stiffness']+damping*matrices['damping']

    def initial(self,replicate,weak=False):
        rng=np.random.default_rng(np.random.SeedSequence([1748,int(self.design.design_id[1:]),replicate]))
        values=self.initial_modes.eigvals; amplitudes=np.exp(rng.normal(size=len(values)))
        if weak: amplitudes[0]*=.01
        coefficients=amplitudes*np.exp(1j*rng.uniform(0,2*np.pi,len(values)))
        physical_q=2*np.real(self.initial_modes.eigvecs@coefficients)
        physical_v=2*np.real(self.initial_modes.eigvecs@(values*coefficients))
        scale=1e-6/max(float(np.sqrt(np.mean((self.sensor@physical_q)**2))),1e-30)
        physical_q*=scale; physical_v*=scale
        states=[]
        for count,matrices in ((self.count,self.reduced),(len(self.omega),self.full)):
            phi=self.basis[:,:count]
            q=solve(matrices['mass'],phi.T@self.M@physical_q,assume_a='pos')
            v=solve(matrices['mass'],phi.T@self.M@physical_v,assume_a='pos')
            states.append(np.r_[self.omega[:count]*q,v])
        return states

    def acquire(self,profile,replicate,forcing,*,coefficient_step=.002):
        times=np.arange(int(round(PROFILES[profile][0][-1]*1000))+1)/1000.
        initial=self.initial(replicate,weak=forcing=='weak_critical_initial')[0]
        return propagate_state(times,initial,lambda t:self.matrix(ratio_at(profile,t)),self.reduced['projection'],
            coefficient_step=coefficient_step,input_vector=self.reduced['force'],forcing=physical_pulse if forcing=='intermittent_physical_pulses' else None)

    def audit(self):
        reduced,full=self.initial(0); duration=.12; times=np.arange(121)/1000.
        records=[]; maximum_frequency=0.
        for ratio in (.9,1.,1.03):
            reference=compute_eigendecomposition(self.solver,ratio*self.ucrit,n_modes=8,sensor_xy=self.sensor_yx,rho=self.design.rho_air,c_sound=self.design.c_air,zeta=self.design.zeta)
            maximum_frequency=max(maximum_frequency,float(max(abs(reference.eigvals.imag))/(2*np.pi)))
            poles,vectors=eig(self.matrix(ratio))
            physical_q=self.basis[:,:self.count]@(vectors[:self.count]/self.omega[:self.count,None])
            visible=(poles.imag>TRANSIENT_FILTER.omega_min)&(_transverse_participation(physical_q,len(self.omega))>=TRANSIENT_FILTER.eta_w_min)
            positive=poles[visible]
            if not len(positive): raise RuntimeError('reduced model has no physical transverse modes')
            critical=positive[np.argmax(positive.real)]
            records.append(dict(ratio=ratio,full_critical=[float(reference.eigvals[0].real),float(reference.eigvals[0].imag)],
                reduced_critical=[float(critical.real),float(critical.imag)],critical_relative_gap=float(abs(critical-reference.eigvals[0])/abs(reference.eigvals[0]))))
        response=[]
        tracks=[('fixed',lambda t:.9,None),('ramp',lambda t:.97+.06*t/duration,None),
                ('forced_ramp',lambda t:.97+.06*t/duration,lambda t:physical_pulse(t+.98))]
        for label,ratio,force in tracks:
            initial_full=full if force is None else np.zeros_like(full)
            initial_reduced=reduced if force is None else np.zeros_like(reduced)
            def derivative(t,x):
                return self.matrix(ratio(t),len(self.omega))@x+self.full['force']*(0. if force is None else force(t))
            reference=solve_ivp(derivative,(0.,duration),initial_full,
                method='Radau',jac=lambda t,x:self.matrix(ratio(t),len(self.omega)),t_eval=times,
                rtol=2e-7,atol=1e-11,max_step=np.inf if force is None else .001)
            if not reference.success: raise RuntimeError(f'full-order trajectory reference failed: {reference.message}')
            truth=self.full['projection']@reference.y
            coarse=propagate_state(times,initial_reduced,lambda t:self.matrix(ratio(t)),self.reduced['projection'],
                coefficient_step=.002,input_vector=self.reduced['force'],forcing=force)
            fine=propagate_state(times,initial_reduced,lambda t:self.matrix(ratio(t)),self.reduced['projection'],
                coefficient_step=.001,input_vector=self.reduced['force'],forcing=force)
            unit=max(float(np.sqrt(np.mean(truth**2))),1e-30)
            response.append(dict(track=label,full_response_relative_rms=float(np.sqrt(np.mean((coarse.response-truth)**2))/unit),
                refined_full_response_relative_rms=float(np.sqrt(np.mean((fine.response-truth)**2))/unit),
                step_halving_relative_rms=float(np.sqrt(np.mean((coarse.response-fine.response)**2))/unit)))
        rom_valid=all(v['full_response_relative_rms']<=.02 and v['step_halving_relative_rms']<=.005 for v in response) and max(v['critical_relative_gap'] for v in records)<=.005
        return dict(poles=records,response=response,rom_valid=bool(rom_valid),retained8_max_frequency_hz=maximum_frequency,
            bandwidth_supported=maximum_frequency<400.,minimum_mach=.9*self.ucrit/self.design.c_air,
            acquisition_supported=bool(rom_valid and maximum_frequency<400. and .9*self.ucrit/self.design.c_air>=2.))


def continuous_noise(raw,profile,forcing,replicate,design_index,kind):
    rng=np.random.default_rng(np.random.SeedSequence([1749,design_index,replicate,list(PROFILES).index(profile),FORCING.index(forcing),NOISE.index(kind)]))
    leading=raw[:,:51]; sigma=float(np.sqrt(np.mean((leading-leading.mean(axis=1,keepdims=True))**2)))
    count=raw.shape[1]; times=np.arange(count)/1000.; channels=raw.shape[0]
    if kind=='clean': noise=np.zeros_like(raw)
    elif kind.startswith('iid'): noise=(.05 if kind=='iid5' else .01)*sigma*rng.normal(size=raw.shape)
    elif kind.startswith('correlated'):
        level=.05 if kind=='correlated5' else .01
        noise=level*sigma*(np.sqrt(.2)*rng.normal(size=raw.shape)+np.sqrt(.8)*rng.normal(size=(1,count)))
    elif kind=='colored_common':
        from scipy.signal import lfilter
        colored=lfilter([np.sqrt(1-.95**2)],[1.,-.95],rng.normal(size=count))
        noise=.01*sigma*(np.sqrt(.2)*rng.normal(size=raw.shape)+np.sqrt(.8)*colored[None,:])
    elif kind=='gain_drift': noise=raw*(.001*times)[None,:]+.01*sigma*rng.normal(size=raw.shape)
    elif kind=='impulses':
        noise=.01*sigma*rng.normal(size=raw.shape)
        for when in (5.,15.,25.,35.):
            start=int(when*1000)
            if start<count: noise[:,start:min(start+5,count)]+=.5*sigma*rng.choice([-1.,1.],size=(channels,1))
    elif kind=='dropout':
        noise=.01*sigma*rng.normal(size=raw.shape)
        for when in (5.,15.,25.,35.):
            start=int(when*1000)
            if start<count: noise[:,start:min(start+500,count)]=np.nan
    else: raise ValueError(kind)
    return noise,sigma
