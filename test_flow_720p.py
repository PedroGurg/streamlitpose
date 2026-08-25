"""
Teste artificial do fluxo completo do app na configuração da Astra Pro:
1280x720 em paisagem, rotacionado 90 graus para retrato (720x1280).

Exercita, sem hardware:
  ETAPA 1 - intrínsecos: várias poses do tabuleiro por câmera
  ETAPA 2 - extrínsecos: captura simultânea das 3 câmeras
  ETAPA 3 - validação: altura/distância recuperadas vs verdade de terreno
  ETAPA 4 - exportação: Calib.toml e round-trip mundo->câmera
  ETAPA 5 - coleta: gravação com barreira, 3 vídeos, checagem de sincronismo

Uso:  python test_flow_720p.py
"""
import os
import shutil
import threading
import time

import cv2
import numpy as np

from charuco_calib import (CharucoConfig, create_board, detect_charuco,
                            sufficient_for_calibration, IntrinsicCalibrator,
                            solve_extrinsics_from_charuco, camera_pose_in_board_frame)
from calib_io import (build_camera_entry, save_calib_toml, axis_swap_matrix,
                       apply_world_transform)
from fake_backend import (World, FakeCameraBackend, look_at, intrinsics_landscape,
                           expected_K_after_rotation, read_barcode)

# ---------------------------------------------------------------------------
# Configuração do rig (mesmos números do documento de campo)
# ---------------------------------------------------------------------------
W_NAT, H_NAT = 1280, 720        # modo RGB da Astra Pro, paisagem
ROTATE = 90                      # rotação de SOFTWARE (deixa a imagem em pé)
ROLL = -90                       # rotação FÍSICA da câmera (põe o eixo longo
                                 # do sensor na vertical -> FOV vertical = 66,1°)
FOV_H = 66.1                     # cenário A; troque para 63.1 para o cenário B
AIM = np.array([0.0, 0.0, 0.95])  # ponto de mira comum

RIG = {
    "cam_01_frontal":  dict(pos=np.array([1.70, 0.00, 1.05])),
    "cam_02_lateral":  dict(pos=np.array([0.00, 1.70, 1.05])),
    "cam_03_diagonal": dict(pos=np.array([1.24, 1.24, 1.75])),
}
CAM_NAMES = list(RIG)

PASS, FAIL = "  OK", "  *** FALHA ***"
results = []


def check(label, cond):
    results.append((label, bool(cond)))
    print(f"    {label}: {'OK' if cond else '*** FALHA ***'}")
    return cond


def banner(txt):
    print()
    print("=" * 70)
    print(txt)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Montagem do mundo virtual
# ---------------------------------------------------------------------------
cfg = CharucoConfig()          # 7x5 @ 8cm, o preset 720p
board, dictionary = create_board(cfg)
bw = cfg.squares_x * cfg.square_length
bh = cfg.squares_y * cfg.square_length

world = World()
world.board_img = board.generateImage((1400, 1000), marginSize=0, borderBits=1)
world.board_size_m = (bw, bh)

K_land = intrinsics_landscape(W_NAT, H_NAT, FOV_H)
K_rot_expected = expected_K_after_rotation(K_land, W_NAT, H_NAT, ROTATE)

backends = {}
truth = {}
for name, spec in RIG.items():
    R, t = look_at(spec["pos"], AIM, roll_deg=ROLL)
    backends[name] = FakeCameraBackend(name, K_land, R, t, W_NAT, H_NAT, world, ROTATE)
    # A rotação de software de 90° equivale a girar o frame da câmera em torno
    # do eixo óptico; a pose no referencial da imagem ROTACIONADA é R_img = Rz(-90) @ R
    a = np.deg2rad(-90.0)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    R_rot = Rz @ R
    truth[name] = dict(pos=spec["pos"], R_w2c=R, t_w2c=t,
                        R_w2c_rot=R_rot, t_w2c_rot=-R_rot @ spec["pos"])

banner("CONFIGURAÇÃO DO TESTE")
print(f"  Sensor            : {W_NAT}x{H_NAT} nativo")
print(f"  Montagem física   : roll de {ROLL}° (eixo longo do sensor na vertical)")
print(f"  Rotação software  : {ROTATE}° -> frame final {H_NAT}x{W_NAT} (retrato)")
print(f"  FOV horizontal    : {FOV_H}°")
print(f"  Tabuleiro         : {cfg.squares_x}x{cfg.squares_y} @ {cfg.square_length*100:.0f}cm "
      f"= {bw*100:.0f}x{bh*100:.0f}cm")
print(f"  fx verdadeiro     : {K_land[0,0]:.2f} px (paisagem)")
print(f"  K esperado após rotação: fx={K_rot_expected[0,0]:.2f} fy={K_rot_expected[1,1]:.2f} "
      f"cx={K_rot_expected[0,2]:.2f} cy={K_rot_expected[1,2]:.2f}")

# ---------------------------------------------------------------------------
# ETAPA 1 - Calibração intrínseca
# ---------------------------------------------------------------------------
banner("ETAPA 1 — CALIBRAÇÃO INTRÍNSECA (por câmera)")

rng = np.random.default_rng(7)
intrinsic_results = {}

for name in CAM_NAMES:
    backend = backends[name]
    calib = None
    n_ok = 0
    attempts = 0
    # Move o tabuleiro perto da câmera, em várias poses, como na aba 2
    cam_pos = truth[name]["pos"]
    while n_ok < 16 and attempts < 120:
        attempts += 1
        # posiciona o tabuleiro entre a câmera e o alvo, com jitter
        f = rng.uniform(0.35, 0.62)
        center = cam_pos + f * (AIM - cam_pos)
        center = center + rng.uniform(-0.16, 0.16, 3)
        # orienta o tabuleiro aproximadamente de frente para a câmera
        n = cam_pos - center
        n = n / np.linalg.norm(n)
        n = n + rng.uniform(-0.30, 0.30, 3)
        n = n / np.linalg.norm(n)
        tmp = np.array([0, 0, 1.0])
        if abs(float(n @ tmp)) > 0.95:
            tmp = np.array([1.0, 0, 0])
        ex = np.cross(tmp, n); ex /= np.linalg.norm(ex)
        ey = np.cross(n, ex)
        world.R_bw = np.column_stack([ex, ey, n])
        world.t_bw = center - world.R_bw @ np.array([bw / 2, bh / 2, 0.0])

        frame = backend.read()
        c, ids, mc, mids = detect_charuco(frame, board, dictionary)
        if not sufficient_for_calibration(ids, min_corners=10):
            continue
        if calib is None:
            h, w = frame.shape[:2]
            calib = IntrinsicCalibrator(board, (w, h))
        calib.add_view(c, ids)
        n_ok += 1

    res = calib.calibrate()
    intrinsic_results[name] = res
    K = res["K"]
    err_fx = abs(K[0, 0] - K_rot_expected[0, 0])
    err_cx = abs(K[0, 2] - K_rot_expected[0, 2])
    err_cy = abs(K[1, 2] - K_rot_expected[1, 2])
    print(f"  {name}:  vistas={n_ok}  image_size={res['image_size']}  "
          f"reproj={res['mean_error_px']:.4f} px")
    print(f"      fx={K[0,0]:8.2f} (esperado {K_rot_expected[0,0]:8.2f}, Δ={err_fx:5.2f})   "
          f"cx={K[0,2]:7.2f} (Δ={err_cx:5.2f})  cy={K[1,2]:7.2f} (Δ={err_cy:5.2f})")
    check(f"{name} image_size em retrato (720x1280)", res["image_size"] == (H_NAT, W_NAT))
    check(f"{name} reproj intrínseca < 0.5 px", res["mean_error_px"] < 0.5)
    check(f"{name} fx recuperado dentro de 1%", err_fx / K_rot_expected[0, 0] < 0.01)

# ---------------------------------------------------------------------------
# ETAPA 2 - Calibração extrínseca (captura simultânea)
# ---------------------------------------------------------------------------
banner("ETAPA 2 — CALIBRAÇÃO EXTRÍNSECA (captura simultânea das 3)")

# Tabuleiro deitado no chão, centralizado na origem do mundo
world.R_bw = np.eye(3)
world.t_bw = np.array([-bw / 2, -bh / 2, 0.0])

extrinsic_results = {}
objp = board.getChessboardCorners()

for name in CAM_NAMES:
    # (a) DETECTABILIDADE: a imagem renderizada é usada só para confirmar que
    #     o tabuleiro no chão é enxergado e quantos cantos aparecem.
    frame = backends[name].read()
    c, ids, mc, mids = detect_charuco(frame, board, dictionary)
    n_c = 0 if ids is None else len(ids)

    # (b) PRECISÃO DE POSE: usa a projeção analítica exata dos pontos de objeto,
    #     com o K JÁ CALIBRADO na etapa 1. Isso testa a cadeia numérica do app
    #     (solvePnP -> inversão de pose -> troca de eixos -> TOML) contra uma
    #     verdade de terreno conhecida, sem depender do renderizador.
    K = intrinsic_results[name]["K"]
    dist = intrinsic_results[name]["dist"]
    R_w2c_true = truth[name]["R_w2c_rot"]
    t_w2c_true = truth[name]["t_w2c_rot"]
    objp_world = objp + np.array([-bw / 2, -bh / 2, 0.0])   # tabuleiro no centro
    rv_t, _ = cv2.Rodrigues(R_w2c_true)
    img_pts = cv2.projectPoints(objp_world, rv_t, t_w2c_true, K, dist)[0]
    ids_all = np.arange(len(objp)).reshape(-1, 1).astype(np.int32)

    rv, tv, err = solve_extrinsics_from_charuco(
        img_pts.reshape(-1, 1, 2).astype(np.float32), ids_all, board, K, dist)
    R_cb, t_cb = camera_pose_in_board_frame(rv, tv)
    extrinsic_results[name] = dict(R_cam_in_board=R_cb, t_cam_in_board=t_cb,
                                    reproj_error_px=err)
    print(f"  {name}: cantos detectados na imagem = {n_c}/24   "
          f"reproj (pose analítica) = {err:.5f} px")
    check(f"{name} detectou 24 cantos no chão (render)", n_c == 24)
    check(f"{name} reproj extrínseca < 1.5 px", err < 1.5)

# ---------------------------------------------------------------------------
# ETAPA 3 - Validação (o que a aba 4 mostraria)
# ---------------------------------------------------------------------------
banner("ETAPA 3 — VALIDAÇÃO CONTRA A VERDADE DE TERRENO")

T = axis_swap_matrix("identity")
# origem do mundo do teste = canto do tabuleiro; a verdade está na origem
# do RIG, então converte para o mesmo referencial
offset = np.array([-bw / 2, -bh / 2, 0.0])   # origem do frame do tabuleiro no mundo

print(f"  {'câmera':18s} {'altura calc':>12s} {'altura real':>12s} "
      f"{'dist calc':>10s} {'dist real':>10s} {'erro 3D':>10s}")
for name in CAM_NAMES:
    if name not in extrinsic_results:
        continue
    ext = extrinsic_results[name]
    R_w, t_w = apply_world_transform(ext["R_cam_in_board"], ext["t_cam_in_board"], T)
    pos_rig = t_w + offset                      # volta ao referencial do rig
    real = truth[name]["pos"]
    h_calc, h_real = pos_rig[2], real[2]
    d_calc = float(np.linalg.norm(pos_rig[:2]))
    d_real = float(np.linalg.norm(real[:2]))
    err3d = float(np.linalg.norm(pos_rig - real))
    print(f"  {name:18s} {h_calc:11.4f}m {h_real:11.4f}m "
          f"{d_calc:9.4f}m {d_real:9.4f}m {err3d*1000:8.2f}mm")
    check(f"{name} erro de posição 3D < 10 mm", err3d < 0.010)

# ---------------------------------------------------------------------------
# ETAPA 4 - Exportação do Calib.toml
# ---------------------------------------------------------------------------
banner("ETAPA 4 — EXPORTAÇÃO DO Calib.toml")

import toml
cameras_out = {}
for name in CAM_NAMES:
    if name not in extrinsic_results:
        continue
    intr = intrinsic_results[name]
    ext = extrinsic_results[name]
    R_w, t_w = apply_world_transform(ext["R_cam_in_board"], ext["t_cam_in_board"], T)
    cameras_out[name] = build_camera_entry(
        name=name, K=intr["K"], dist=intr["dist"], image_size=intr["image_size"],
        R_cam_in_world=R_w, t_cam_in_world=t_w)

out_toml = "test_out/Calib.toml"
os.makedirs("test_out", exist_ok=True)
save_calib_toml(cameras_out, out_toml)
loaded = toml.load(out_toml)
check("Calib.toml tem as 3 câmeras", len(loaded) == 3)

for name, entry in loaded.items():
    R_w2c, _ = cv2.Rodrigues(np.array(entry["rotation"]))
    t_w2c = np.array(entry["translation"])
    # o centro da câmera no mundo deve mapear para a origem da câmera
    ext = extrinsic_results[name]
    R_w, t_w = apply_world_transform(ext["R_cam_in_board"], ext["t_cam_in_board"], T)
    residual = np.linalg.norm(R_w2c @ t_w + t_w2c)
    print(f"  {name}: size={entry['size']}  round-trip mundo->câmera = {residual:.2e}")
    check(f"{name} round-trip mundo->câmera exato", residual < 1e-9)
    check(f"{name} size em retrato no TOML", entry["size"] == [H_NAT, W_NAT])

# ---------------------------------------------------------------------------
# ETAPA 5 - Coleta de dados (mesma lógica de threads/barreira da aba 5)
# ---------------------------------------------------------------------------
banner("ETAPA 5 — COLETA DE DADOS (gravação com barreira)")

world.mode = "person"
world.person_height = 1.75
trial_dir = "test_out/coleta/trial_01/videos"
if os.path.exists("test_out/coleta"):
    shutil.rmtree("test_out/coleta")
os.makedirs(trial_dir, exist_ok=True)

target_fps = 30
duration_s = 2
n_frames_target = int(duration_s * target_fps)
frame_interval = 1.0 / target_fps

# --- mesma sequência da aba 5 ---
first_frames = {}
world.frame_counter = 0
for name in CAM_NAMES:
    first_frames[name] = backends[name].read()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writers = {}
for name in CAM_NAMES:
    h, w = first_frames[name].shape[:2]
    writers[name] = cv2.VideoWriter(os.path.join(trial_dir, f"{name}.mp4"),
                                     fourcc, target_fps, (w, h))
    writers[name].write(first_frames[name])

remaining = n_frames_target - 1
n_workers = len(CAM_NAMES)
barrier_grab = threading.Barrier(n_workers + 1)
barrier_done = threading.Barrier(n_workers + 1)
stop_flag = threading.Event()
shared_frames = {}
errors = {}

# ESTE é o ponto do bug corrigido: backends resolvidos ANTES das threads
backends_for_threads = {name: backends[name] for name in CAM_NAMES}


def camera_worker(name, backend):
    try:
        writer = writers[name]
    except Exception as e:
        errors[name] = f"falha ao iniciar worker: {e}"
        stop_flag.set()
        barrier_grab.abort()
        barrier_done.abort()
        return
    for _ in range(remaining):
        try:
            barrier_grab.wait(timeout=10)
        except threading.BrokenBarrierError:
            return
        try:
            frame = backend.read()
            if frame is None:
                raise RuntimeError("sem frame")
            writer.write(frame)
            shared_frames[name] = frame
        except Exception as e:
            errors[name] = str(e)
            stop_flag.set()
        finally:
            try:
                barrier_done.wait(timeout=10)
            except threading.BrokenBarrierError:
                return
        if stop_flag.is_set():
            return


threads = [threading.Thread(target=camera_worker, args=(n, backends_for_threads[n]),
                            daemon=True) for n in CAM_NAMES]
for t in threads:
    t.start()

n_written = 1
t_start = time.time()
aborted = False
for i in range(remaining):
    target_t = t_start + (i + 1) * frame_interval
    now = time.time()
    if target_t > now:
        time.sleep(target_t - now)
    # o "mundo" avança ANTES do disparo: pessoa se move e contador incrementa
    world.frame_counter = (i + 1) % 256
    phase = (i + 1) / n_frames_target
    world.person_pos = np.array([0.30 * np.sin(2 * np.pi * phase), 0.0,
                                  0.30 * max(0.0, np.sin(np.pi * phase))])
    try:
        barrier_grab.wait(timeout=10)
        barrier_done.wait(timeout=10)
    except threading.BrokenBarrierError:
        aborted = True
        break
    if stop_flag.is_set():
        aborted = True
        break
    n_written += 1

for t in threads:
    t.join(timeout=2)
for wr in writers.values():
    wr.release()

total_time = time.time() - t_start
measured_fps = n_written / total_time if total_time > 0 else 0.0

print(f"  abortado          : {aborted}   erros: {errors if errors else 'nenhum'}")
print(f"  quadros gravados  : {n_written}/{n_frames_target}")
print(f"  fps medido        : {measured_fps:.1f} (alvo {target_fps})")
check("gravação não abortou", not aborted)
check("gravou todos os quadros", n_written == n_frames_target)

# --- verificação dos arquivos ---
counts, sizes = {}, {}
barcodes = {}
for name in CAM_NAMES:
    path = os.path.join(trial_dir, f"{name}.mp4")
    cap = cv2.VideoCapture(path)
    seq, n = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        if sizes.get(name) is None:
            sizes[name] = (fr.shape[1], fr.shape[0])
        # o barcode foi desenhado no frame em PAISAGEM, antes da rotação de
        # software; desfaz a rotação para ler na orientação original
        seq.append(read_barcode(cv2.rotate(fr, cv2.ROTATE_90_COUNTERCLOCKWISE)))
    cap.release()
    counts[name] = n
    barcodes[name] = seq
    print(f"  {name}: {n} quadros, {sizes[name][0]}x{sizes[name][1]} px, "
          f"{os.path.getsize(path)/1024:.0f} KB")
    check(f"{name} vídeo em retrato 720x1280", sizes[name] == (H_NAT, W_NAT))

check("os 3 vídeos têm o mesmo número de quadros",
      len(set(counts.values())) == 1)

# --- sincronismo: mesmo contador no mesmo índice, nas 3 câmeras ---
n_min = min(len(v) for v in barcodes.values())
mismatches = 0
for i in range(n_min):
    vals = {barcodes[n][i] for n in CAM_NAMES}
    if len(vals) != 1:
        mismatches += 1
print(f"  quadros comparados: {n_min}   divergências entre câmeras: {mismatches}")
check("as 3 câmeras gravaram o mesmo instante em cada índice", mismatches == 0)

# ---------------------------------------------------------------------------
banner("RESUMO")
n_pass = sum(1 for _, ok in results if ok)
n_tot = len(results)
for label, ok in results:
    if not ok:
        print(f"  FALHOU: {label}")
print(f"\n  {n_pass}/{n_tot} verificações passaram")
print("  " + ("TODAS AS VERIFICAÇÕES PASSARAM" if n_pass == n_tot
              else "*** HÁ FALHAS -- veja acima ***"))
