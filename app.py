"""
App de calibração multi-câmera (ChArUco) para o pipeline Pose2Sim.

Fluxo:
  1) Setup: conectar as 3 câmeras (webcam para testar / Orbbec em campo).
  2) Intrínsecos: para cada câmera, capturar várias vistas do tabuleiro
     movido pelo quadro e calibrar. Alvo: erro médio < 0,5 px.
  3) Extrínsecos: com o tabuleiro parado no centro, capturar UM instante
     simultâneo nas 3 câmeras e resolver a pose de cada uma por PnP.
     Alvo: erro de reprojeção compatível com < 1 cm no volume de trabalho.
  4) Validação e exportação: comparar com medidas de trena, plotar as
     câmeras em 3D, exportar Calib.toml.
  5) Coleta de dados: gravar, com as câmeras já calibradas, um vídeo por
     câmera (mesma duração/fps) para alimentar a etapa de pose 2D e
     triangulação do Pose2Sim.

Rodar com:  streamlit run app.py
"""

import glob
import os
import time
import threading
import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from camera_backend import make_backend
from charuco_calib import (
    CharucoConfig, create_board, detect_charuco, draw_detection,
    sufficient_for_calibration, IntrinsicCalibrator,
    solve_extrinsics_from_charuco, camera_pose_in_board_frame,
)
from calib_io import build_camera_entry, save_calib_toml, axis_swap_matrix, apply_world_transform

st.set_page_config(page_title="Calibração multi-câmera — ChArUco", layout="wide")

CAM_NAMES = ["cam_01_frontal", "cam_02_lateral", "cam_03_diagonal"]

# ---------------------------------------------------------------------------
# Estado
# ---------------------------------------------------------------------------
if "backends" not in st.session_state:
    st.session_state.backends = {name: None for name in CAM_NAMES}
if "board_cfg" not in st.session_state:
    st.session_state.board_cfg = CharucoConfig()
if "intrinsic_calibrators" not in st.session_state:
    st.session_state.intrinsic_calibrators = {name: None for name in CAM_NAMES}
if "intrinsic_results" not in st.session_state:
    st.session_state.intrinsic_results = {name: None for name in CAM_NAMES}
if "extrinsic_results" not in st.session_state:
    st.session_state.extrinsic_results = {name: None for name in CAM_NAMES}
if "manual_measurements" not in st.session_state:
    st.session_state.manual_measurements = {
        name: {"height_m": 0.0, "distance_m": 0.0} for name in CAM_NAMES
    }

INTRINSIC_TARGET_PX = 0.5
EXTRINSIC_TARGET_PX = 1.5  # ver nota na aba de validação sobre a conversão px -> mm

# ---------------------------------------------------------------------------
# Sidebar: configuração do tabuleiro e das câmeras
# ---------------------------------------------------------------------------
st.sidebar.header("Configuração do tabuleiro ChArUco")
cfg = st.session_state.board_cfg
cfg.squares_x = st.sidebar.number_input("Quadrados em X", 3, 20, cfg.squares_x)
cfg.squares_y = st.sidebar.number_input("Quadrados em Y", 3, 20, cfg.squares_y)
cfg.square_length = st.sidebar.number_input("Lado do quadrado (m)", 0.01, 0.5, cfg.square_length, format="%.3f")
cfg.marker_length = st.sidebar.number_input("Lado do marcador ArUco (m)", 0.01, 0.5, cfg.marker_length, format="%.3f")
cfg.dict_name = st.sidebar.selectbox(
    "Dicionário ArUco", ["DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_250"],
    index=["DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_250"].index(cfg.dict_name),
)
cfg.legacy_pattern = st.sidebar.checkbox(
    "Padrão legado (tabuleiros calib.io / pré-OpenCV 4.6)", cfg.legacy_pattern,
    help="Ative se o tabuleiro foi gerado pelo calib.io ou outra ferramenta externa. "
         "Sem isso, os marcadores são detectados mas os cantos ChArUco não são "
         "interpolados (fica em 0 mesmo com o tabuleiro bem enquadrado).",
)
board, dictionary = create_board(cfg)

st.sidebar.header("Câmeras")
backend_kind = st.sidebar.radio("Fonte de vídeo", ["opencv", "orbbec"], horizontal=True,
                                 help="Use 'opencv' para testar com webcams antes de ir a campo.")
img_w = st.sidebar.number_input(
    "Largura nativa do sensor (px)", 320, 3840, 1280,
    help="Resolução PEDIDA à câmera, antes de rotacionar. Para o backend "
         "'orbbec' precisa bater EXATAMENTE com um perfil suportado pelo "
         "sensor (o erro de conexão lista os perfis disponíveis, com formato, "
         "se errar). A Astra Pro entrega RGB a 1280x720; 640x480 costuma "
         "estar disponível também.",
)
img_h = st.sidebar.number_input(
    "Altura nativa do sensor (px)", 240, 2160, 720,
    help="Idem, altura nativa. 720 para o modo RGB da Astra Pro (16:9); "
         "480 para o modo 640x480 (4:3).",
)
rotate_deg = st.sidebar.selectbox(
    "Rotação (pós-captura)", [0, 90, 180, 270], index=0,
    help="Sensores 4:3 montados de lado para enquadrar o corpo inteiro em "
         "retrato precisam de 90° (ou 270°, dependendo do lado do giro). "
         "A rotação é aplicada em software depois da captura — não muda o "
         "que é pedido ao sensor acima.",
)
st.sidebar.caption(
    f"Frame final após rotação: "
    f"{img_h if rotate_deg in (90, 270) else img_w} × "
    f"{img_w if rotate_deg in (90, 270) else img_h} px."
)

st.sidebar.header("Armazenamento")
capture_dir = st.sidebar.text_input(
    "Pasta de capturas", "capturas",
    help="Todo frame usado na calibração é salvo aqui em PNG. Permite "
         "recalibrar depois sem remontar o rig.")

cam_indices = {}
for i, name in enumerate(CAM_NAMES):
    cam_indices[name] = st.sidebar.number_input(f"Índice / device_id — {name}", 0, 10, i, key=f"idx_{name}")

if st.sidebar.button("Conectar câmeras"):
    for name in CAM_NAMES:
        old = st.session_state.backends[name]
        if old is not None:
            old.release()
        try:
            b = make_backend(backend_kind, cam_indices[name], img_w, img_h, rotate_deg=rotate_deg)
            b.connect()
            st.session_state.backends[name] = b
            st.sidebar.success(f"{name}: conectada")
        except Exception as e:
            st.session_state.backends[name] = None
            st.sidebar.error(f"{name}: falha — {e}")

# ---------------------------------------------------------------------------
# Abas
# ---------------------------------------------------------------------------
tab_setup, tab_intrinsic, tab_extrinsic, tab_validate, tab_collect = st.tabs(
    ["1. Setup", "2. Intrínsecos", "3. Extrínsecos", "4. Validação e exportação", "5. Coleta de dados"]
)

# ---------------------------------------------------------------------------
# Aba 1: Setup — preview rápido de cada câmera
# ---------------------------------------------------------------------------
with tab_setup:
    st.subheader("Preview das câmeras")
    st.caption("Confirme enquadramento e foco antes de calibrar. Clique de novo para atualizar.")
    cols = st.columns(3)
    for col, name in zip(cols, CAM_NAMES):
        with col:
            st.markdown(f"**{name}**")
            backend = st.session_state.backends[name]
            if backend is None:
                st.warning("Não conectada.")
                continue
            frame = backend.read()
            if frame is None:
                st.error("Sem frame.")
                continue
            frame_rgb = frame[:, :, ::-1]
            st.image(frame_rgb, use_container_width=True)

# ---------------------------------------------------------------------------
# Aba 2: Calibração intrínseca (por câmera, independente)
# ---------------------------------------------------------------------------
with tab_intrinsic:
    st.subheader("Calibração intrínseca — uma câmera de cada vez")
    st.caption(
        "Mova o tabuleiro cobrindo todo o quadro, inclusive as bordas. "
        "Capture pelo menos 12–15 vistas variando ângulo e posição."
    )

    active_cam = st.selectbox("Câmera ativa", CAM_NAMES, key="intrinsic_active_cam")
    backend = st.session_state.backends[active_cam]

    if st.session_state.intrinsic_calibrators[active_cam] is None:
        st.session_state.intrinsic_calibrators[active_cam] = None  # criado após 1º frame (precisa do tamanho)

    col_live, col_info = st.columns([2, 1])

    with col_live:
        live_slot = st.empty()

    with col_info:
        n_views = 0
        calibrator = st.session_state.intrinsic_calibrators[active_cam]
        if calibrator is not None:
            n_views = calibrator.n_views()
        st.metric("Vistas capturadas", n_views)
        capture_btn = st.button("📸 Capturar vista atual", key=f"cap_{active_cam}")
        reset_btn = st.button("🗑️ Reiniciar vistas desta câmera", key=f"reset_{active_cam}")
        run_calib_btn = st.button("▶️ Rodar calibração intrínseca", key=f"run_{active_cam}")

    if reset_btn:
        st.session_state.intrinsic_calibrators[active_cam] = None
        st.session_state.intrinsic_results[active_cam] = None
        st.rerun()

    if backend is not None:
        frame = backend.read()
        if frame is not None:
            h, w = frame.shape[:2]
            charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco(frame, board, dictionary)
            overlay = draw_detection(frame, charuco_corners, charuco_ids, marker_corners, marker_ids)
            live_slot.image(overlay[:, :, ::-1], use_container_width=True)

            detected_n = 0 if charuco_ids is None else len(charuco_ids)
            col_info.caption(f"Cantos ChArUco detectados agora: {detected_n}")

            if capture_btn:
                if sufficient_for_calibration(charuco_ids):
                    if st.session_state.intrinsic_calibrators[active_cam] is None:
                        st.session_state.intrinsic_calibrators[active_cam] = IntrinsicCalibrator(board, (w, h))
                    st.session_state.intrinsic_calibrators[active_cam].add_view(charuco_corners, charuco_ids)
                    idx = st.session_state.intrinsic_calibrators[active_cam].n_views()
                    save_dir = os.path.join(capture_dir, "intrinsecos", active_cam)
                    os.makedirs(save_dir, exist_ok=True)
                    png_path = os.path.join(save_dir, f"view_{idx:03d}.png")
                    cv2.imwrite(png_path, frame)
                    st.success(f"Vista adicionada ({detected_n} cantos) — salva em {png_path}")
                else:
                    st.warning("Detecção insuficiente nesta vista — não foi adicionada.")
        else:
            st.error("Sem frame desta câmera.")
    else:
        st.warning("Conecte a câmera na aba Setup primeiro.")

    if run_calib_btn:
        calibrator = st.session_state.intrinsic_calibrators[active_cam]
        if calibrator is None or calibrator.n_views() < 4:
            st.error("Capture pelo menos 4-6 vistas antes de calibrar (recomendado: 12+).")
        else:
            with st.spinner("Calibrando..."):
                result = calibrator.calibrate()
            st.session_state.intrinsic_results[active_cam] = result
            st.success(f"Calibrado. Erro médio de reprojeção: {result['mean_error_px']:.3f} px")

    with st.expander("♻️ Recalibrar intrínsecos a partir de imagens salvas (sem câmera)"):
        st.caption(
            "Relê os PNGs de "
            f"`{os.path.join(capture_dir, 'intrinsecos', active_cam)}` e refaz a "
            "calibração. Use se quiser mudar parâmetros do tabuleiro ou "
            "reprocessar depois de desmontar o rig.")
        if st.button("Recalibrar do disco", key=f"reload_intr_{active_cam}"):
            d = os.path.join(capture_dir, "intrinsecos", active_cam)
            files = sorted(glob.glob(os.path.join(d, "*.png")))
            if not files:
                st.error(f"Nenhum PNG encontrado em {d}")
            else:
                calib_r, n_ok, n_bad = None, 0, 0
                for f in files:
                    im = cv2.imread(f)
                    if im is None:
                        n_bad += 1
                        continue
                    cc, ci, _, _ = detect_charuco(im, board, dictionary)
                    if not sufficient_for_calibration(ci):
                        n_bad += 1
                        continue
                    if calib_r is None:
                        hh, ww = im.shape[:2]
                        calib_r = IntrinsicCalibrator(board, (ww, hh))
                    calib_r.add_view(cc, ci)
                    n_ok += 1
                if calib_r is None or calib_r.n_views() < 4:
                    st.error(f"Vistas válidas insuficientes ({n_ok} de {len(files)}).")
                else:
                    res_r = calib_r.calibrate()
                    st.session_state.intrinsic_calibrators[active_cam] = calib_r
                    st.session_state.intrinsic_results[active_cam] = res_r
                    st.success(
                        f"Recalibrado de {n_ok} imagens ({n_bad} descartadas). "
                        f"Erro médio: {res_r['mean_error_px']:.3f} px")

    st.divider()
    st.markdown("**Resumo de todas as câmeras**")
    summary_rows = []
    for name in CAM_NAMES:
        res = st.session_state.intrinsic_results[name]
        if res is None:
            summary_rows.append({"câmera": name, "vistas": "-", "erro médio (px)": "-", "status": "não calibrada"})
        else:
            n = len(res["per_view_error_px"])
            err = res["mean_error_px"]
            status = "✅ dentro do alvo" if err < INTRINSIC_TARGET_PX else "⚠️ acima do alvo"
            summary_rows.append({"câmera": name, "vistas": n, "erro médio (px)": f"{err:.3f}", "status": status})
    st.table(summary_rows)
    st.caption(f"Alvo: erro médio de reprojeção < {INTRINSIC_TARGET_PX} px (Pose2Sim docs).")

# ---------------------------------------------------------------------------
# Aba 3: Calibração extrínseca — captura simultânea das 3 câmeras
# ---------------------------------------------------------------------------
with tab_extrinsic:
    st.subheader("Calibração extrínseca — captura simultânea")
    st.caption(
        "Posicione o tabuleiro parado no centro, visível pelas 3 câmeras, e capture. "
        "Repita em 2-3 poses (chão, inclinado, elevado) para robustez — "
        "cada captura é registrada como uma 'pose do tabuleiro' abaixo."
    )

    all_ready = all(st.session_state.intrinsic_results[n] is not None for n in CAM_NAMES)
    if not all_ready:
        st.warning("Calibre os intrínsecos das 3 câmeras antes de prosseguir (aba 2).")

    if "extrinsic_poses" not in st.session_state:
        st.session_state.extrinsic_poses = []  # lista de dicts: {cam_name: (R,t,err)}

    capture_sync_btn = st.button("📸 Capturar instante simultâneo (as 3 câmeras)", disabled=not all_ready)

    if capture_sync_btn:
        pose_result = {}
        captured_frames = {}
        preview_cols = st.columns(3)
        for col, name in zip(preview_cols, CAM_NAMES):
            backend = st.session_state.backends[name]
            frame = backend.read()
            if frame is None:
                col.error(f"{name}: sem frame")
                continue
            captured_frames[name] = frame
            charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco(frame, board, dictionary)
            overlay = draw_detection(frame, charuco_corners, charuco_ids, marker_corners, marker_ids)
            col.image(overlay[:, :, ::-1], caption=name, use_container_width=True)

            if not sufficient_for_calibration(charuco_ids, min_corners=8):
                col.warning("Detecção insuficiente — descarte esta captura.")
                continue

            K = st.session_state.intrinsic_results[name]["K"]
            dist = st.session_state.intrinsic_results[name]["dist"]
            rvec, tvec, err_px = solve_extrinsics_from_charuco(charuco_corners, charuco_ids, board, K, dist)
            R_cam_in_board, t_cam_in_board = camera_pose_in_board_frame(rvec, tvec)
            pose_result[name] = {
                "R_cam_in_board": R_cam_in_board,
                "t_cam_in_board": t_cam_in_board,
                "reproj_error_px": err_px,
            }
            col.caption(f"Erro de reprojeção: {err_px:.3f} px")

        if len(pose_result) == 3:
            pose_n = len(st.session_state.extrinsic_poses) + 1
            pdir = os.path.join(capture_dir, "extrinsecos", f"pose_{pose_n:02d}")
            os.makedirs(pdir, exist_ok=True)
            for nm, fr in captured_frames.items():
                cv2.imwrite(os.path.join(pdir, f"{nm}.png"), fr)
            st.caption(f"Frames desta pose salvos em `{pdir}`")
            st.session_state.extrinsic_poses.append(pose_result)
            st.success(f"Pose #{len(st.session_state.extrinsic_poses)} registrada para as 3 câmeras.")
        else:
            st.error("Nem todas as câmeras detectaram o tabuleiro nesta captura — não foi registrada.")

    with st.expander("♻️ Recarregar poses extrínsecas salvas (sem câmera)"):
        st.caption(f"Relê as pastas `pose_XX` de `{os.path.join(capture_dir, 'extrinsecos')}`.")
        if st.button("Recarregar do disco", disabled=not all_ready):
            base = os.path.join(capture_dir, "extrinsecos")
            pose_dirs = sorted(glob.glob(os.path.join(base, "pose_*")))
            loaded_poses, skipped = [], []
            for pd_ in pose_dirs:
                pr = {}
                for name in CAM_NAMES:
                    fp = os.path.join(pd_, f"{name}.png")
                    im = cv2.imread(fp) if os.path.exists(fp) else None
                    if im is None:
                        continue
                    cc, ci, _, _ = detect_charuco(im, board, dictionary)
                    if not sufficient_for_calibration(ci, min_corners=8):
                        continue
                    Kk = st.session_state.intrinsic_results[name]["K"]
                    dd = st.session_state.intrinsic_results[name]["dist"]
                    rv2, tv2, e2 = solve_extrinsics_from_charuco(cc, ci, board, Kk, dd)
                    Rr, tt = camera_pose_in_board_frame(rv2, tv2)
                    pr[name] = dict(R_cam_in_board=Rr, t_cam_in_board=tt,
                                     reproj_error_px=e2)
                if len(pr) == 3:
                    loaded_poses.append(pr)
                else:
                    skipped.append(os.path.basename(pd_))
            if loaded_poses:
                st.session_state.extrinsic_poses = loaded_poses
                msg = f"{len(loaded_poses)} pose(s) recarregada(s)."
                if skipped:
                    msg += f" Ignoradas (nem todas as 3 câmeras detectaram): {skipped}"
                st.success(msg)
            else:
                st.error(f"Nenhuma pose completa encontrada em {base}")

    st.divider()
    n_poses = len(st.session_state.extrinsic_poses)
    st.metric("Poses simultâneas capturadas", n_poses)

    if n_poses > 0:
        pose_idx = st.selectbox(
            "Usar qual pose como referência do mundo?",
            list(range(n_poses)),
            format_func=lambda i: f"Pose #{i+1}",
        )
        if st.button("✅ Fixar extrínsecos a partir desta pose"):
            chosen = st.session_state.extrinsic_poses[pose_idx]
            for name in CAM_NAMES:
                st.session_state.extrinsic_results[name] = chosen[name]
            st.success("Extrínsecos fixados. Vá para a aba de validação.")

        st.caption(
            "Se capturou mais de uma pose, você pode comparar manualmente os R/t "
            "entre poses como checagem de consistência antes de fixar."
        )

# ---------------------------------------------------------------------------
# Aba 4: Validação e exportação
# ---------------------------------------------------------------------------
with tab_validate:
    st.subheader("Validação contra medidas manuais")
    st.caption(
        "Compare a altura e a distância que a calibração devolveu com o que "
        "você mediu com trena no rig. Divergência grande aponta erro na "
        "calibração ou no referencial do mundo, não necessariamente no tripé."
    )

    axis_mode = st.radio(
        "Transformação de eixos do mundo",
        ["identity", "z_to_y_up"],
        horizontal=True,
        help="'identity': tabuleiro já está com Z para cima e é isso que você quer. "
             "'z_to_y_up': converte para a convenção Y-para-cima do OpenSim.",
    )
    T = axis_swap_matrix(axis_mode)

    all_extrinsics_ready = all(st.session_state.extrinsic_results[n] is not None for n in CAM_NAMES)

    rows = []
    world_positions = {}
    for name in CAM_NAMES:
        ext = st.session_state.extrinsic_results[name]
        if ext is None:
            rows.append({"câmera": name, "altura calc. (m)": "-", "distância calc. (m)": "-",
                         "erro reproj. (px)": "-", "status": "extrínseco pendente"})
            continue

        R_world, t_world = apply_world_transform(ext["R_cam_in_board"], ext["t_cam_in_board"], T)
        world_positions[name] = t_world

        height_calc = t_world[2] if axis_mode == "identity" else t_world[1]
        distance_calc = float(np.linalg.norm(t_world[:2] if axis_mode == "identity" else t_world[[0, 2]]))

        meas = st.session_state.manual_measurements[name]
        delta_h = abs(height_calc - meas["height_m"]) if meas["height_m"] > 0 else None
        delta_d = abs(distance_calc - meas["distance_m"]) if meas["distance_m"] > 0 else None

        status_bits = []
        if ext["reproj_error_px"] > EXTRINSIC_TARGET_PX:
            status_bits.append("⚠️ reproj. alta")
        if delta_h is not None and delta_h > 0.10:
            status_bits.append("⚠️ altura diverge >10cm")
        if delta_d is not None and delta_d > 0.10:
            status_bits.append("⚠️ distância diverge >10cm")
        status = " / ".join(status_bits) if status_bits else "✅ ok"

        rows.append({
            "câmera": name,
            "altura calc. (m)": f"{height_calc:.3f}",
            "distância calc. (m)": f"{distance_calc:.3f}",
            "erro reproj. (px)": f"{ext['reproj_error_px']:.3f}",
            "status": status,
        })

    st.markdown("**Medidas manuais (trena) — preencha para habilitar a comparação**")
    meas_cols = st.columns(3)
    for col, name in zip(meas_cols, CAM_NAMES):
        with col:
            st.markdown(f"*{name}*")
            st.session_state.manual_measurements[name]["height_m"] = st.number_input(
                "Altura do centro óptico (m)", 0.0, 3.0,
                st.session_state.manual_measurements[name]["height_m"], key=f"h_{name}"
            )
            st.session_state.manual_measurements[name]["distance_m"] = st.number_input(
                "Distância horizontal ao centro (m)", 0.0, 5.0,
                st.session_state.manual_measurements[name]["distance_m"], key=f"d_{name}"
            )

    st.table(rows)
    st.caption(
        f"Alvo de reprojeção extrínseca: < {EXTRINSIC_TARGET_PX} px. "
        "Este é um proxy em pixels — a correspondência exata com milímetros no mundo "
        "depende da distância e da focal; use a comparação com a trena acima como "
        "checagem independente em unidades reais."
    )

    if world_positions:
        st.subheader("Geometria do rig (visual)")
        fig = go.Figure()
        xs = [p[0] for p in world_positions.values()]
        ys = [p[1] for p in world_positions.values()]
        zs = [p[2] for p in world_positions.values()]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="markers+text",
            text=list(world_positions.keys()), marker=dict(size=6),
        ))
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                                    marker=dict(size=8, color="red"), name="origem (tabuleiro)"))
        fig.update_layout(scene=dict(aspectmode="data"), height=500,
                           margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Confira: as posições relativas devem bater com os azimutes e distâncias planejados (0°, 90°, 45°).")

    st.divider()
    st.subheader("Exportar Calib.toml")
    st.caption(
        "Confira as chaves geradas contra um Calib.toml de exemplo do próprio "
        "Pose2Sim antes de usar em produção — ver nota em calib_io.py."
    )
    export_path = st.text_input("Caminho de saída", "/mnt/user-data/outputs/Calib.toml")

    if st.button("💾 Exportar", disabled=not all_extrinsics_ready):
        cameras_out = {}
        for name in CAM_NAMES:
            intr = st.session_state.intrinsic_results[name]
            ext = st.session_state.extrinsic_results[name]
            R_world, t_world = apply_world_transform(ext["R_cam_in_board"], ext["t_cam_in_board"], T)
            entry = build_camera_entry(
                name=name, K=intr["K"], dist=intr["dist"], image_size=intr["image_size"],
                R_cam_in_world=R_world, t_cam_in_world=t_world,
            )
            cameras_out[name] = entry
        save_calib_toml(cameras_out, export_path)
        st.success(f"Exportado para {export_path}")

# ---------------------------------------------------------------------------
# Aba 5: Coleta de dados — grava um vídeo sincronizado por câmera
# ---------------------------------------------------------------------------
with tab_collect:
    st.subheader("Coleta de dados — gravação sincronizada")
    st.caption(
        "Grava um vídeo por câmera, com a mesma duração e fps alvo, prontos "
        "para a etapa de pose 2D e triangulação do Pose2Sim. Exporte o "
        "Calib.toml na aba anterior antes de processar esses vídeos lá."
    )

    all_connected = all(st.session_state.backends[n] is not None for n in CAM_NAMES)
    if not all_connected:
        st.warning("Conecte as 3 câmeras na aba Setup antes de gravar.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        output_dir = st.text_input("Pasta de saída", "coleta")
    with col_b:
        trial_name = st.text_input("Nome do trial", "trial_01")
    with col_c:
        duration_s = st.number_input("Duração (s)", 1, 300, 10)

    target_fps = st.number_input("FPS alvo", 5, 60, 30)

    record_btn = st.button("🔴 Gravar trial", disabled=not all_connected)

    if record_btn:
        trial_dir = os.path.join(output_dir, trial_name, "videos")
        os.makedirs(trial_dir, exist_ok=True)

        first_frames = {}
        for name in CAM_NAMES:
            frame = st.session_state.backends[name].read()
            if frame is None:
                st.error(f"{name}: sem frame — gravação abortada antes de começar.")
                st.stop()
            first_frames[name] = frame

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writers = {}
        for name in CAM_NAMES:
            h, w = first_frames[name].shape[:2]
            path = os.path.join(trial_dir, f"{name}.mp4")
            writers[name] = cv2.VideoWriter(path, fourcc, target_fps, (w, h))
            writers[name].write(first_frames[name])

        preview_cols = st.columns(3)
        preview_slots = {name: col.empty() for col, name in zip(preview_cols, CAM_NAMES)}
        for name in CAM_NAMES:
            preview_slots[name].image(first_frames[name][:, :, ::-1], caption=name, use_container_width=True)
        progress = st.progress(0.0)
        status = st.empty()

        n_frames_target = int(duration_s * target_fps)
        frame_interval = 1.0 / target_fps

        # Grava os quadros restantes com as 3 câmeras em threads separadas,
        # sincronizadas por uma barreira: nenhuma thread chama .read() antes
        # que TODAS estejam prontas para o mesmo quadro, evitando o atraso
        # que existia ao ler uma câmera de cada vez em sequência.
        remaining = n_frames_target - 1
        n_workers = len(CAM_NAMES)
        barrier_grab = threading.Barrier(n_workers + 1)
        barrier_done = threading.Barrier(n_workers + 1)
        stop_flag = threading.Event()
        shared_frames = {}
        errors = {}

        # IMPORTANTE: st.session_state NÃO é acessível de threads secundárias --
        # elas não herdam o ScriptRunContext e recebem um estado vazio, o que
        # levantaria AttributeError e mataria a thread antes da barreira.
        # Os backends são resolvidos AQUI, na thread principal, e passados por
        # argumento para os workers.
        backends_for_threads = {name: st.session_state.backends[name] for name in CAM_NAMES}

        def camera_worker(name, backend):
            try:
                writer = writers[name]
            except Exception as e:
                # Qualquer falha antes do laço deixaria a barreira travada por
                # 10s e abortaria com 'errors' vazio. Registra e quebra as
                # barreiras explicitamente para falhar rápido e com mensagem.
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

        threads = [
            threading.Thread(target=camera_worker, args=(name, backends_for_threads[name]), daemon=True)
            for name in CAM_NAMES
        ]
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
            for name in CAM_NAMES:
                preview_slots[name].image(shared_frames[name][:, :, ::-1], caption=name, use_container_width=True)
            progress.progress(min(n_written / n_frames_target, 1.0))
            status.caption(f"{n_written}/{n_frames_target} quadros")

        for t in threads:
            t.join(timeout=2)

        for w_ in writers.values():
            w_.release()

        total_time = time.time() - t_start
        measured_fps = n_written / total_time if total_time > 0 else 0.0

        if aborted:
            err_msg = "; ".join(f"{name}: {msg}" for name, msg in errors.items())
            status.error(f"Trial abortado em {n_written} quadros. {err_msg}")
            st.warning(f"Trial parcial salvo em {trial_dir} ({n_written} quadros).")
        else:
            st.success(
                f"Trial salvo em {trial_dir} — {n_written} quadros por câmera, "
                f"~{measured_fps:.1f} fps medido (alvo: {target_fps} fps), "
                "capturados em threads sincronizadas por barreira."
            )
        st.caption(
            "Se o fps medido ficou bem abaixo do alvo, a leitura das câmeras é o "
            "gargalo (reduza resolução ou fps alvo) — mesmo sincronizadas, os "
            "quadros vão sair mais espaçados do que o previsto nesse caso."
        )
