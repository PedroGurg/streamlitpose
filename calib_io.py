"""
Exportação da calibração para Calib.toml (formato Pose2Sim/AniPose).

IMPORTANTE: confira as chaves exatas geradas aqui contra um Calib.toml de
exemplo gerado pelo próprio Pose2Sim (Pose2Sim.calibration() no projeto
Demo) antes de importar em produção -- o formato pode variar entre
versões do pacote. Esta função reproduz a estrutura documentada no
momento da implementação (matriz 3x3, distorção, rotação como vetor de
Rodrigues, translação em metros).
"""

from __future__ import annotations
import numpy as np
import toml


def axis_swap_matrix(mode: str) -> np.ndarray:
    """
    Transformações comuns entre "tabuleiro deitado no chão, Z para cima"
    e a convenção Y-para-cima usada pelo OpenSim/Pose2Sim.

    'identity'   : nenhuma troca (use se já confirmou que Z do tabuleiro
                   aponta para cima e X/Y estão como deseja)
    'z_to_y_up'  : troca Z<->Y, mantendo destreza (Z do tabuleiro vira o
                   novo Y "para cima")
    """
    if mode == "identity":
        return np.eye(3)
    if mode == "z_to_y_up":
        # X_novo = X_antigo ; Y_novo = Z_antigo ; Z_novo = -Y_antigo
        return np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ], dtype=float)
    raise ValueError(f"Modo de troca de eixos desconhecido: {mode}")


def apply_world_transform(R_cam_in_board, t_cam_in_board, T: np.ndarray):
    """
    Aplica a matriz de troca de eixos T (3x3, ortogonal) ao referencial
    do mundo. R e t passam a ser expressos no novo referencial.
    """
    R_new = T @ R_cam_in_board
    t_new = T @ t_cam_in_board
    return R_new, t_new


def build_camera_entry(name: str, K: np.ndarray, dist: np.ndarray,
                        image_size: tuple[int, int],
                        R_cam_in_world: np.ndarray, t_cam_in_world: np.ndarray) -> dict:
    """
    Monta o bloco de uma câmera no formato esperado.

    Observação sobre convenção: Pose2Sim/OpenCV descrevem a pose da CENA
    em relação à CÂMERA (rotation/translation "world-to-camera"), não o
    contrário. Se R_cam_in_world / t_cam_in_world foram calculados como
    "câmera no referencial do mundo" (camera-to-world, que é o que
    camera_pose_in_board_frame devolve), inverta antes de gravar:

        R_w2c = R_cam_in_world.T
        t_w2c = -R_w2c @ t_cam_in_world

    Esta função já faz essa inversão internamente.
    """
    R_w2c = R_cam_in_world.T
    t_w2c = -R_w2c @ t_cam_in_world
    rvec_w2c, _ = cv2_rodrigues_safe(R_w2c)

    width, height = image_size
    return {
        "name": name,
        "size": [width, height],
        "matrix": K.tolist(),
        "distortions": np.asarray(dist).flatten().tolist(),
        "rotation": rvec_w2c.flatten().tolist(),
        "translation": t_w2c.flatten().tolist(),
        "fisheye": False,
    }


def cv2_rodrigues_safe(R: np.ndarray):
    import cv2
    rvec, jac = cv2.Rodrigues(R)
    return rvec, jac


def save_calib_toml(cameras: dict, path: str) -> None:
    """
    cameras: dict {cam_name: camera_entry_dict} conforme build_camera_entry
    """
    out = {}
    for cam_name, entry in cameras.items():
        out[cam_name] = entry
    with open(path, "w") as f:
        toml.dump(out, f)
