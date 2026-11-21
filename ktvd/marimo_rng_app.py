import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium", app_title="КТвД · Интерактивный генератор случайных чисел (NO₃⁻ / pH)")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import scipy.stats as st
    import matplotlib.pyplot as plt
    return mo, np, pd, st, plt


@app.cell
def _(mo):
    mo.md(
        """
        # Интерактивный генератор случайных чисел (Лабораторная работа 01 · КТвД)
        Моделирование гидрохимических показателей подземных вод штата Телангана (`NO₃⁻` и `pH`) с использованием **метода полярных координат Марсальи**, **метода Бокса–Мюллера** и **метода обратной функции (квантильного преобразования)**.
        """
    )
    return


@app.cell
def _(mo):
    variable_select = mo.ui.dropdown(
        options={
            "Концентрация нитратов NO₃⁻ (мг/л, LogNormal)": "NO3",
            "Водородный показатель pH (ед. pH, Normal)": "pH",
        },
        value="Концентрация нитратов NO₃⁻ (мг/л, LogNormal)",
        label="Исследуемая случайная величина:",
    )
    method_select = mo.ui.dropdown(
        options={
            "Метод полярных координат (Марсалья)": "polar",
            "Метод Бокса–Мюллера": "box_muller",
            "Метод обратной функции (квантильное преобразование)": "inv_cdf",
        },
        value="Метод полярных координат (Марсалья)",
        label="Алгоритм датчика случайных чисел:",
    )
    sample_size = mo.ui.slider(start=100, stop=5000, step=100, value=1106, label="Объём выборки генератора (N):")
    seed_input = mo.ui.number(start=1, stop=99999, step=1, value=42, label="Зерно ГСЧ (Random Seed):")
    bins_slider = mo.ui.slider(start=15, stop=60, step=5, value=30, label="Количество интервалов гистограммы:")

    controls = mo.vstack([
        mo.hstack([variable_select, method_select], justify="start", gap=2),
        mo.hstack([sample_size, seed_input, bins_slider], justify="start", gap=2),
    ])
    controls
    return bins_slider, controls, method_select, sample_size, seed_input, variable_select


@app.cell
def _(np, pd, st):
    df = pd.read_csv("telangana_groundwater_2018_2020.csv")

    def generate_standard_normal(n: int, method: str, rng: np.random.Generator) -> np.ndarray:
        if method == "polar":
            out = []
            while len(out) < n:
                u1, u2 = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
                s = u1 * u1 + u2 * u2
                if 0.0 < s < 1.0:
                    factor = np.sqrt(-2.0 * np.log(s) / s)
                    out.extend([u1 * factor, u2 * factor])
            return np.array(out[:n])
        elif method == "box_muller":
            m = (n + 1) // 2
            u1 = rng.uniform(1e-12, 1.0, size=m)
            u2 = rng.uniform(0.0, 1.0, size=m)
            r = np.sqrt(-2.0 * np.log(u1))
            theta = 2.0 * np.pi * u2
            z = np.concatenate([r * np.cos(theta), r * np.sin(theta)])
            return z[:n]
        else:  # inv_cdf
            u = rng.uniform(1e-9, 1.0 - 1e-9, size=n)
            return st.norm.ppf(u)

    return df, generate_standard_normal


@app.cell
def _(bins_slider, df, generate_standard_normal, method_select, mo, np, pd, plt, sample_size, seed_input, st, variable_select):
    rng = np.random.default_rng(int(seed_input.value))
    var_code = variable_select.value
    method_code = method_select.value
    n_sim = int(sample_size.value)

    if var_code == "NO3":
        real_data = df["NO3"].dropna().values
        log_data = np.log(real_data)
        mu, sigma = float(np.mean(log_data)), float(np.std(log_data, ddof=1))
        z = generate_standard_normal(n_sim, method_code, rng)
        sim_data = np.exp(mu + sigma * z)
        unit_label = "Концентрация NO₃⁻ (мг/л)"
        x_limit = (0, 450)
    else:
        real_data = df["pH"].dropna().values
        mu, sigma = float(np.mean(real_data)), float(np.std(real_data, ddof=1))
        z = generate_standard_normal(n_sim, method_code, rng)
        sim_data = mu + sigma * z
        unit_label = "Водородный показатель pH (ед. pH)"
        x_limit = (6.0, 10.5)

    ks_stat, ks_p = st.kstest(real_data, sim_data)

    metrics_table = pd.DataFrame({
        "Метрика": ["Объём выборки (N)", "Среднее (Mean)", "Медиана (Median)", "Ст. отклонение (Std)", "25% квантиль (Q1)", "75% квантиль (Q3)", "KS-тест (D / p-value)"],
        "Реальные данные": [
            f"{len(real_data)}",
            f"{np.mean(real_data):.3f}",
            f"{np.median(real_data):.3f}",
            f"{np.std(real_data, ddof=1):.3f}",
            f"{np.quantile(real_data, 0.25):.3f}",
            f"{np.quantile(real_data, 0.75):.3f}",
            "—",
        ],
        "Сгенерированные данные (ГСЧ)": [
            f"{len(sim_data)}",
            f"{np.mean(sim_data):.3f}",
            f"{np.median(sim_data):.3f}",
            f"{np.std(sim_data, ddof=1):.3f}",
            f"{np.quantile(sim_data, 0.25):.3f}",
            f"{np.quantile(sim_data, 0.75):.3f}",
            f"D = {ks_stat:.4f} (p = {ks_p:.3f})",
        ],
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(real_data, bins=bins_slider.value, range=x_limit, density=True, alpha=0.45, color="#1D3557", edgecolor="white", label="Реальные данные")
    axes[0].hist(sim_data, bins=bins_slider.value, range=x_limit, density=True, alpha=0.45, color="#E76F51", edgecolor="white", label="Генератор (ГСЧ)")
    axes[0].set_title("Сравнение гистограмм плотности")
    axes[0].set_xlabel(unit_label)
    axes[0].set_ylabel("Плотность вероятности")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].legend()

    q_grid = np.linspace(0.01, 0.99, 99)
    qr, qs = np.quantile(real_data, q_grid), np.quantile(sim_data, q_grid)
    axes[1].scatter(qr, qs, color="#2A9D8F", alpha=0.8, s=28, edgecolor="white")
    lim_hi = max(qr.max(), qs.max())
    lim_lo = min(qr.min(), qs.min())
    axes[1].plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="#E76F51", lw=2, ls="--", label="Эталон y = x")
    axes[1].set_title("Q-Q график: реальные данные vs ГСЧ")
    axes[1].set_xlabel(f"Квантили реальных данных ({unit_label})")
    axes[1].set_ylabel(f"Квантили ГСЧ ({unit_label})")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].legend()
    plt.tight_layout()

    mo.vstack([
        mo.ui.table(metrics_table, selection=None),
        fig,
    ])
    return axes, fig, ks_p, ks_stat, lim_hi, lim_lo, log_data, method_code, metrics_table, mu, n_sim, q_grid, qr, qs, real_data, rng, sigma, sim_data, unit_label, var_code, x_limit, z


if __name__ == "__main__":
    app.run()
