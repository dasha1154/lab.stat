# Lab 01 — Applied Statistics (InfoChemistry × KTvD)

## Структура проекта
- `labwork.ipynb` — основной ноутбук исследования (Sprint 1: `TDS`, `F`, `Rainfall` + шаблон Sprint 2).
- `data/water/` — исходные наборы данных мониторинга качества подземных вод (2018–2020).
- `ktvd_extra_task/` — **дополнительное задание для КТвД**:
  - `telangana_groundwater_2018_2020.csv` — объединённый датасет ($N = 1106$ наблюдений).
  - `ktvd_extra_analysis.ipynb` — исследование двух случайных величин одной природы (`pH` и `NO₃⁻`): доказательство стохастичности, EDA, нормализация, доказательство зависимости и верификация ГСЧ.
  - `marimo_rng_app.py` — интерактивное веб-приложение на **Marimo** для генератора случайных чисел.

## Запуск интерактивного приложения Marimo
```bash
pip install marimo numpy pandas scipy matplotlib
cd ktvd_extra_task
marimo edit marimo_rng_app.py
```
