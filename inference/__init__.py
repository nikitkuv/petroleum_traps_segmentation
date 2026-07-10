"""Пакет инференса: прогон обученной модели на новых CPS-гридах.

run_inference живёт в inference.predict (импортируется явно из run_inference.py),
чтобы лёгкие подмодули (tiling/postprocess) не подтягивали torch/smp/wandb.
"""
