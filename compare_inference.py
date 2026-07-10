"""
Сравнение двух режимов предсказания ловушек на одних и тех же картах:

  1) evaluate (пертайл) — evaluation.evaluate_on_raw_cps: модель гоняется по тайлам,
     метрики считаются по каждому тайлу и микро-агрегируются (пиксели в зонах
     перекрытия тайлов учитываются 2–4 раза);
  2) inference (склейка) — inference.run_inference: те же тайлы + склейка в единую
     карту (усреднение перекрытий), метрики по полной карте (каждый пиксель один раз).

Цель: проверить, не ухудшает ли склейка тайлов качество предсказаний относительно
опорного пертайлового режима evaluate_model.py.

Обе пайплайн видят ОДИНАКОВЫЙ вход (временная папка _cmp_in с копиями выбранных карт),
одну модель и один порог. Для чистоты сравнения:
  - evaluate запускается с min_traps_pixels=0 (покрытие = все тайлы, как у инференса);
  - инференс — без постобработки (min_area=0, fill_holes=False, TTA=False);
  => единственная содержательная разница — склейка/усреднение перекрытий и метод
     подсчёта метрик. Подробности и интерпретация — в docs/COMPARISON.md.

Запуск:  python compare_inference.py
Артефакты: logs/test_inference/{evaluate,inference}/{ts}/ (визуализации + метрики).
"""
import os
import shutil

from settings import settings
from evaluation.evaluate_on_raw_cps import evaluate_on_raw_cps
from inference.predict import run_inference
from utils.dataset_utils import extract_base_horizon


CHECKPOINT = "checkpoints/v4_cps_tiles_640x448_faults_e-50_bs-4_lr-0.0003(0.1)_wd-0.005_rlp.pth"

# Карты для сравнения (имена горизонтов, как в CPS-файлах). Правьте под свой сценарий.
# Набор намеренно разный по размеру: от 1 тайла до ~12 (много перекрытий).
MAPS = [
    "Ach3-2-1_toptop1",   # маленькая, ~1 тайл
    "H150_toptop1",       # ~2 тайла, есть разломы
    "CS1_bottop1",        # val-горизонт, ~2 тайла, есть разломы
    "U9_10_kolltop1",     # крупная, ~12 тайлов — стресс-тест склейки/перекрытий
]

CPS_SRC = settings.CPS_SOURCE_DIR
WORK_DIR = "./_cmp_in"   # временная папка с копиями карт (пересоздаётся, в конце удаляется)
USE_FAULTS = settings.USE_FAULTS


def build_input_folder(maps, dst):
    """Копирует нужные CPS-гриды выбранных карт во временную папку (оригиналы не трогаем)."""
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for h in maps:
        for prefix in ("x_structuralNOisoline_", "x_faults_", "y_traps_"):
            src = os.path.join(CPS_SRC, prefix + h)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(dst, prefix + h))


def main():
    # Обе пайплайн пишут визуализации/метрики под logs/test_inference/{evaluate,inference}/
    settings.LOGS_TEST_VIZ_DIR = "./logs/test_inference/evaluate"
    settings.LOGS_INFER_VIZ_DIR = "./logs/test_inference/inference"

    build_input_folder(MAPS, WORK_DIR)

    print("\n############ EVALUATE (пертайл, min_traps=0) ############")
    eval_res = evaluate_on_raw_cps(
        checkpoint_path=CHECKPOINT, cps_dir=WORK_DIR, horizon_prefixes=[],
        use_faults=USE_FAULTS, min_traps_pixels=0, save_viz=True, save_metrics=True,
    )

    print("\n############ INFERENCE (склейка, постобработка OFF) ############")
    inf_res = run_inference(
        checkpoint_path=CHECKPOINT, cps_dir=WORK_DIR, horizon_prefixes=[],
        use_faults=USE_FAULTS, min_trap_area_px=0, fill_holes=False, tta=False,
    )

    _print_comparison(eval_res, inf_res, MAPS)

    shutil.rmtree(WORK_DIR, ignore_errors=True)


def _print_comparison(eval_res, inf_res, maps):
    print("\n\n========== COMPARE: evaluate (per-tile) vs inference (stitched) ==========")
    header = (f"{'map':<20} {'method':<7} {'tiles':>6} {'Dice':>7} {'IoU':>7} "
              f"{'Recall':>7} {'Prec':>7} {'F1':>7} {'FP%':>6} {'FN%':>6}")
    print(header)
    print("-" * 96)
    for full in maps:
        base = extract_base_horizon(full)
        e = eval_res['per_horizon_metrics'].get(base, {})
        ires = inf_res['per_horizon'].get(full, {})
        im = ires.get('metrics') or {}
        if e:
            print(f"{full:<20} {'eval':<7} {e.get('n_tiles','-'):>6} "
                  f"{e['dice']:>7.3f} {e['iou']:>7.3f} {e['recall']:>7.3f} "
                  f"{e['precision']:>7.3f} {e['f1']:>7.3f} {e['fp_area']:>6.3f} {e['fn_area']:>6.3f}")
        else:
            print(f"{full:<20} {'eval':<7}  нет метрик (base={base})")
        if im:
            print(f"{'':<20} {'infer':<7} {ires.get('n_tiles','-'):>6} "
                  f"{im.get('dice', 0):>7.3f} {im.get('iou', 0):>7.3f} {im.get('recall', 0):>7.3f} "
                  f"{im.get('precision', 0):>7.3f} {im.get('f1', 0):>7.3f} "
                  f"{im.get('fp_area', 0):>6.3f} {im.get('fn_area', 0):>6.3f}")
        print()

    ea = eval_res['aggregated_metrics']
    ia = inf_res['aggregated_metrics_macro']
    print("-" * 96)
    print(f"{'AGGREGATED':<20} {'eval':<7} {'':>6} "
          f"{ea['dice']:>7.3f} {ea['iou']:>7.3f} {ea['recall']:>7.3f} "
          f"{ea['precision']:>7.3f} {ea['f1']:>7.3f} {ea['fp_area']:>6.3f} {ea['fn_area']:>6.3f}")
    if ia.get('n_horizons_with_gt', 0):
        print(f"{'AGGREGATED (macro)':<20} {'infer':<7} {'':>6} "
              f"{ia.get('dice', 0):>7.3f} {ia.get('iou', 0):>7.3f} {ia.get('recall', 0):>7.3f} "
              f"{ia.get('precision', 0):>7.3f} {ia.get('f1', 0):>7.3f} "
              f"{ia.get('fp_area', 0):>6.3f} {ia.get('fn_area', 0):>6.3f}")
    print("=" * 96)
    print(f"\neval артефакты:      {os.path.join(settings.LOGS_TEST_VIZ_DIR, eval_res['run_timestamp'])}")
    print(f"inference артефакты: {inf_res['viz_dir']}")
    print(f"inference CPS-гриды: {inf_res['cps_out_root']}")


if __name__ == "__main__":
    main()
