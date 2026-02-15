import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
import sys
try:
    from rich.console import Console
    from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
    HAVE_RICH = True
except Exception:
    HAVE_RICH = False

G = 6.67430e-11
AU = 1.496e11
M_SUN = 1.9891e30
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
STYLES = [
    ("#1f77b4", "-", "o"),
    ("#d62728", "--", "s"),
    ("#2ca02c", "-.", "^"),
    ("#9467bd", ":", "D"),
    ("#ff7f0e", (0, (5, 1)), "v"),
    ("#17becf", (0, (3, 1, 1, 1)), "P"),
]
MODES = ("ballistic", "accretor")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 22.5,
        "axes.titlesize": 27,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
console = Console() if HAVE_RICH else None


def out(*args, **kwargs):
    if console:
        console.print(*args, **kwargs)
    else:
        print(*args)


def make_progress():
    if HAVE_RICH:
        return Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
    class P:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def add_task(self, *_ , total=0):
            return {"total": total}
        def update(self, *_ , **__):
            return None
        def advance(self, *_ , **__):
            return None
    return P()


def save_pdf(fig, stem):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def styled(ax, x, y, i, label=None, lw=2, markevery=None):
    c, ls, mk = STYLES[i % len(STYLES)]
    ax.plot(
        x,
        y,
        color=c,
        linestyle=ls,
        linewidth=lw,
        marker=mk,
        markevery=markevery,
        markersize=5,
        markerfacecolor="white",
        markeredgecolor=c,
        markeredgewidth=1.2,
        label=label,
    )


@dataclass
class Body:
    m: float
    r: np.ndarray
    v: np.ndarray

    def __post_init__(self):
        self.r = np.asarray(self.r, float)
        self.v = np.asarray(self.v, float)

    def L(self):
        return self.m * (self.r[0] * self.v[1] - self.r[1] * self.v[0])

    def KE(self):
        return 0.5 * self.m * np.dot(self.v, self.v)


class Binary:
    def __init__(self, donor, accretor, mode):
        if mode not in MODES:
            raise ValueError(f"Unknown prescription: {mode}")
        self.d, self.a, self.mode = donor, accretor, mode

    def sep(self):
        return np.linalg.norm(self.a.r - self.d.r)

    def mtot(self):
        return self.d.m + self.a.m

    def q(self):
        return self.d.m / self.a.m

    def L(self):
        return self.d.L() + self.a.L()

    def E(self):
        return self.d.KE() + self.a.KE() - G * self.d.m * self.a.m / self.sep()

    def a_osc(self):
        r = self.sep()
        v = np.linalg.norm(self.a.v - self.d.v)
        e = 0.5 * v * v - G * self.mtot() / r
        return np.inf if abs(e) < 1e-16 else -G * self.mtot() / (2 * e)

    def grav(self):
        rv = self.a.r - self.d.r
        r = np.linalg.norm(rv)
        h = rv / r
        return G * self.a.m / r**2 * h, -G * self.d.m / r**2 * h

    def step(self, dt):
        a1, a2 = self.grav()
        self.d.r += self.d.v * dt + 0.5 * a1 * dt * dt
        self.a.r += self.a.v * dt + 0.5 * a2 * dt * dt
        b1, b2 = self.grav()
        self.d.v += 0.5 * (a1 + b1) * dt
        self.a.v += 0.5 * (a2 + b2) * dt

    def transfer(self, dm):
        if dm <= 0:
            return
        dm = min(dm, self.d.m * (1 - 1e-6))
        if self.mode == "ballistic":
            p = self.a.m * self.a.v + dm * self.d.v
            self.d.m -= dm
            self.a.m += dm
            self.a.v = p / self.a.m
        else:
            p = self.d.m * self.d.v - dm * self.a.v
            self.d.m -= dm
            self.a.m += dm
            self.d.v = p / self.d.m

    def recenter(self):
        m = self.mtot()
        rc = (self.d.m * self.d.r + self.a.m * self.a.r) / m
        vc = (self.d.m * self.d.v + self.a.m * self.a.v) / m
        self.d.r -= rc
        self.a.r -= rc
        self.d.v -= vc
        self.a.v -= vc


def run_sim(M1, M2, a0, mode, f=0.15, n_orbits=50, n_trans_per_orbit=10, steps_per_orbit=10000, progress=None, task=None):
    M = M1 + M2
    v = np.sqrt(G * M / a0)
    sys = Binary(
        Body(M1, [-a0 * M2 / M, 0], [0, -v * M2 / M]),
        Body(M2, [a0 * M1 / M, 0], [0, v * M1 / M]),
        mode,
    )
    P = 2 * np.pi * np.sqrt(a0**3 / (G * M))
    dt = P / steps_per_orbit
    total = int(n_orbits * steps_per_orbit)
    interval = max(1, steps_per_orbit // n_trans_per_orbit)
    rec = max(1, steps_per_orbit // 10)
    n_events = max(1, int(n_orbits * n_trans_per_orbit))
    target = f * M1
    dm = target / n_events
    moved = done = 0
    n_rec = total // rec + 1
    t = np.empty(n_rec)
    s = np.empty(n_rec)
    L = np.empty(n_rec)
    E = np.empty(n_rec)
    q = np.empty(n_rec)
    a = np.empty(n_rec)
    idx = 0
    t[0] = 0.0
    s[0] = sys.sep()
    L[0] = sys.L()
    E[0] = sys.E()
    q[0] = sys.q()
    a[0] = sys.a_osc()
    L0, a_init, E0 = L[0], a[0], E[0]
    tick = max(1, total // 300)
    advanced = 0
    step_fn = sys.step
    transfer_fn = sys.transfer
    recenter_fn = sys.recenter
    sep_fn = sys.sep
    L_fn = sys.L
    E_fn = sys.E
    q_fn = sys.q
    a_fn = sys.a_osc
    for step in range(1, total + 1):
        step_fn(dt)
        if step % interval == 0 and done < n_events:
            d = min(dm, target - moved, sys.d.m * 0.999)
            if d > 0:
                transfer_fn(d)
                recenter_fn()
                moved += d
                done += 1
        if step % rec == 0:
            idx += 1
            tt = step * dt
            t[idx] = tt
            s[idx] = sep_fn()
            L[idx] = L_fn()
            E[idx] = E_fn()
            q[idx] = q_fn()
            a[idx] = a_fn()
        if progress and task and step % tick == 0:
            progress.update(task, advance=tick)
            advanced += tick
    if progress and task and advanced < total:
        progress.update(task, advance=total - advanced)
    used = slice(0, idx + 1)
    return {
        "times": t[used],
        "separations": s[used],
        "angular_momenta": L[used],
        "energies": E[used],
        "mass_ratios": q[used],
        "semi_major_axes": a[used],
        "L0": L0,
        "a0": a_init,
        "E0": E0,
        "P_orb": P,
        "prescription": mode,
        "q0": M1 / M2,
        "f_transfer": f,
    }


def classical_prediction(q0, f):
    return 1 / (((1 - f) * (1 + q0 * f)) ** 2)


def compare_prescriptions(q0, f=0.15, n_orbits=50):
    M2 = M_SUN
    M1 = q0 * M2
    a0 = AU
    out = {}
    total_steps = int(n_orbits * 10000)
    with make_progress() as p:
        for mode in MODES:
            task = p.add_task(f"Simulating {mode} at q={q0}", total=total_steps)
            out[mode] = run_sim(M1, M2, a0, mode, f, n_orbits, progress=p, task=task)
    return out, classical_prediction(q0, f)


def generate_core_figures(q0=0.5, f=0.15, n_orbits=50):
    results, a_class = compare_prescriptions(q0, f, n_orbits)
    tag = f"q{q0:.1f}_f{f:.2f}".replace(".", "p")
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, mode in enumerate(MODES):
        r = results[mode]
        styled(ax, r["times"] / r["P_orb"], r["semi_major_axes"] / r["a0"], i, mode.capitalize(), markevery=8)
    ax.axhline(a_class, color="0.4", linestyle=":", linewidth=1.5, label=f"Classical: {a_class:.3f}")
    ax.axhline(1.0, color="0.6", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Time [orbital periods]")
    ax.set_ylabel(r"$a/a_0$")
    ax.set_title(f"Orbital Evolution (q={q0}, f={f:.2f})")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_pdf(fig, f"fig2_orbital_evolution_{tag}")
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, mode in enumerate(MODES):
        r = results[mode]
        y = (r["angular_momenta"] - r["L0"]) / r["L0"] * 100
        styled(ax, r["times"] / r["P_orb"], y, i, mode.capitalize(), markevery=8)
    ax.axhline(0.0, color="0.4", linestyle=":", linewidth=1.5, label="Conserved")
    ax.set_xlabel("Time [orbital periods]")
    ax.set_ylabel(r"$\Delta L/L_0$ [%]")
    ax.set_title(f"Angular Momentum Evolution (q={q0}, f={f:.2f})")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_pdf(fig, f"fig3_angular_momentum_{tag}")
    return results


def plot_eta_parameter_space(q_values=(0.5, 1.0, 2.0)):
    fig, ax = plt.subplots(figsize=(8, 6))
    eta = np.linspace(0, 1, 500)
    y_min = y_max = 0.0
    for i, q in enumerate(q_values):
        y = 1 - eta * (1 + 1 / q)
        styled(ax, eta, y, i, rf"$q={q}$", lw=2.1, markevery=70)
        ax.plot([q / (1 + q)], [0], marker=STYLES[i % len(STYLES)][2], color=STYLES[i % len(STYLES)][0], markersize=6, markerfacecolor="white", markeredgewidth=1.2)
        y_min = min(y_min, y.min())
        y_max = max(y_max, y.max())
    pad = 0.15 * (y_max - y_min if y_max > y_min else 1)
    ax.fill_between(eta, 0, y_max + pad, color="0.92")
    ax.fill_between(eta, y_min - pad, 0, color="0.82")
    ax.axhline(0, color="black", linestyle=":", linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel(r"Prescription parameter $\eta$")
    ax.set_ylabel(r"$(\Delta L/L)/\delta m$")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.2)
    save_pdf(fig, "fig1_eta_parameter_space")


def scan_mass_ratios(q_values, f=0.15, n_orbits=50):
    rows = []
    with make_progress() as p:
        task = p.add_task("Scanning mass ratios", total=len(q_values) * 2)
        for q0 in q_values:
            M2 = M_SUN
            M1 = q0 * M2
            a0 = AU
            row = {"q0": q0}
            for mode in MODES:
                r = run_sim(M1, M2, a0, mode, f, n_orbits, steps_per_orbit=5000)
                row[f"dL_L_{mode}"] = (r["angular_momenta"][-1] - r["L0"]) / r["L0"]
                row[f"af_a0_{mode}"] = r["semi_major_axes"][-1] / r["a0"]
                p.advance(task)
            row["af_a0_classical"] = classical_prediction(q0, f)
            rows.append(row)
    return rows


def plot_parameter_scan(q_values, f=0.15, n_orbits=50):
    rows = scan_mass_ratios(q_values, f, n_orbits)
    q = [r["q0"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    styled(ax1, q, [r["af_a0_ballistic"] for r in rows], 0, "Ballistic", markevery=1)
    styled(ax1, q, [r["af_a0_accretor"] for r in rows], 1, "Accretor", markevery=1)
    ax1.plot(q, [r["af_a0_classical"] for r in rows], color="0.3", linestyle=":", linewidth=1.6, label="Classical")
    ax1.axhline(1, color="0.6", linewidth=0.8)
    ax1.set_xlabel(r"$q_0=M_1/M_2$")
    ax1.set_ylabel(r"$a_f/a_0$")
    ax1.set_title(f"Orbital Response vs q (f={f:.2f})")
    ax1.grid(True, alpha=0.25)
    ax1.legend()
    styled(ax2, q, [r["dL_L_ballistic"] * 100 for r in rows], 0, "Ballistic", markevery=1)
    styled(ax2, q, [r["dL_L_accretor"] * 100 for r in rows], 1, "Accretor", markevery=1)
    ax2.axhline(0, color="0.3", linestyle=":", linewidth=1.4)
    ax2.set_xlabel(r"$q_0=M_1/M_2$")
    ax2.set_ylabel(r"$\Delta L/L_0$ [%]")
    ax2.set_title("Angular Momentum Change vs q")
    ax2.grid(True, alpha=0.25)
    ax2.legend()
    save_pdf(fig, "fig4_parameter_scan")
    return rows


def generate_fig5(f=0.15, q_values=(0.5, 1.0, 2.0, 3.0)):
    eta = np.linspace(0, 1, 500)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    with make_progress() as p:
        task = p.add_task("Building Figure 5", total=len(q_values))
        for i, q0 in enumerate(q_values):
            a = 1 / ((1 - f) ** (2 * (1 - eta)) * (1 + f * q0) ** (2 * eta))
            dL = (1 + f * q0) ** (1 - eta) * (1 - f) ** eta - 1
            eta_c = q0 / (1 + q0)
            a_c = 1 / ((1 - f) ** (2 * (1 - eta_c)) * (1 + f * q0) ** (2 * eta_c))
            styled(ax1, eta, a, i, rf"$q_0={q0}$", lw=1.9, markevery=70)
            styled(ax2, eta, dL, i, rf"$q_0={q0}$", lw=1.9, markevery=70)
            c, _, mk = STYLES[i % len(STYLES)]
            ax1.plot([eta_c], [a_c], marker=mk, color=c, markersize=7, markerfacecolor="white", markeredgewidth=1.2)
            ax2.plot([eta_c], [0], marker=mk, color=c, markersize=7, markerfacecolor="white", markeredgewidth=1.2)
            p.advance(task)
    ax1.axhline(1, color="0.4", linestyle=":", linewidth=1)
    ax2.axhline(0, color="0.4", linestyle=":", linewidth=1)
    ax1.set_xlim(0, 1)
    ax2.set_xlim(0, 1)
    ax1.set_xlabel(r"Prescription parameter $\eta$")
    ax2.set_xlabel(r"Prescription parameter $\eta$")
    ax1.set_ylabel(r"$a_f/a_0$")
    ax2.set_ylabel(r"$\Delta L/L_0$")
    ax1.set_title(f"Final separation ratio (f={f:.2f})")
    ax2.set_title(f"Angular momentum change (f={f:.2f})")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    save_pdf(fig, "fig5_eta_family_scan")


def run_all():
    plot_eta_parameter_space()
    generate_core_figures(q0=0.5, f=0.15, n_orbits=50)
    generate_core_figures(q0=2.0, f=0.15, n_orbits=50)
    plot_parameter_scan([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5], f=0.15, n_orbits=50)
    generate_fig5()
    pdfs = sorted(p.name for p in FIG_DIR.glob("*.pdf"))
    out("\nGenerated PDFs:")
    for name in pdfs:
        out(f" - {name}")


if __name__ == "__main__":
    run_all()
