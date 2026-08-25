"""
Núcleo de calibração com ChArUco.

Cobre:
- criação do tabuleiro
- detecção de cantos/marcadores num frame
- acúmulo de vistas e calibração intrínseca (por câmera, offline)
- calibração extrínseca por PnP a partir de uma única detecção
  (usado na captura simultânea das 3 câmeras vendo o mesmo tabuleiro)
- erro de reprojeção em pixels

Convenção de eixos: o referencial do mundo é o do próprio tabuleiro na
pose em que ele foi capturado como "referência" (tipicamente deitado no
chão). Eixo Z do tabuleiro aponta para fora do plano impresso. Se o
tabuleiro estiver deitado com a face impressa para cima, Z do tabuleiro
já aponta para cima -- mas CONFIRME visualmente antes de assumir isso
(ver app.py, aba de validação, plot 3D).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2
from cv2 import aruco


@dataclass
class CharucoConfig:
    """
    Padrão: 7x5 quadrados de 8 cm = tabuleiro de 56 x 40 cm.

    Esse tamanho foi escolhido por teste: com as câmeras a ~1,70 m e FOV
    horizontal de ~63 graus, um tabuleiro de 4 cm/quadrado rende apenas
    ~12-16 px por quadrado e FALHA na câmera 3 (elevada, 25 graus) --
    zero marcadores detectados. Com 8 cm/quadrado sobem para ~29-35 px e
    os 24 cantos são detectados nas três posições do rig.

    Se aproximar as câmeras, é possível reduzir; se afastar, aumente.
    Regra prática: mirar 30 px ou mais por quadrado na distância de trabalho.
    """
    squares_x: int = 7
    squares_y: int = 5
    square_length: float = 0.08   # metros
    marker_length: float = 0.06   # metros
    dict_name: str = "DICT_5X5_100"
    legacy_pattern: bool = False   # True para tabuleiros gerados pelo calib.io (numeração pré-OpenCV 4.6)


def get_dictionary(dict_name: str):
    return aruco.getPredefinedDictionary(getattr(aruco, dict_name))


def create_board(cfg: CharucoConfig):
    dictionary = get_dictionary(cfg.dict_name)
    board = aruco.CharucoBoard(
        (cfg.squares_x, cfg.squares_y),
        cfg.square_length,
        cfg.marker_length,
        dictionary,
    )
    board.setLegacyPattern(cfg.legacy_pattern)
    return board, dictionary


def detect_charuco(frame_bgr: np.ndarray, board, dictionary):
    """
    Retorna (charuco_corners, charuco_ids, marker_corners, marker_ids).
    Qualquer um pode vir None se nada for detectado.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    detector_params = aruco.DetectorParameters()
    aruco_detector = aruco.ArucoDetector(dictionary, detector_params)
    marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)

    if marker_ids is None or len(marker_ids) == 0:
        return None, None, marker_corners, marker_ids

    charuco_detector = aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(
        gray, markerCorners=marker_corners, markerIds=marker_ids
    )
    return charuco_corners, charuco_ids, marker_corners, marker_ids


def draw_detection(frame_bgr: np.ndarray, charuco_corners, charuco_ids, marker_corners, marker_ids):
    out = frame_bgr.copy()
    if marker_ids is not None and len(marker_ids) > 0:
        aruco.drawDetectedMarkers(out, marker_corners, marker_ids)
    if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 0:
        aruco.drawDetectedCornersCharuco(out, charuco_corners, charuco_ids, (0, 255, 0))
    return out


def sufficient_for_calibration(charuco_ids, min_corners: int = 6) -> bool:
    return charuco_ids is not None and len(charuco_ids) >= min_corners


class IntrinsicCalibrator:
    """Acumula vistas (cantos ChArUco detectados em vários frames) de UMA câmera."""

    def __init__(self, board, image_size: tuple[int, int]):
        self.board = board
        self.image_size = image_size  # (width, height)
        self.all_corners: list[np.ndarray] = []
        self.all_ids: list[np.ndarray] = []

    def add_view(self, charuco_corners, charuco_ids) -> None:
        self.all_corners.append(charuco_corners)
        self.all_ids.append(charuco_ids)

    def n_views(self) -> int:
        return len(self.all_corners)

    def calibrate(self):
        """
        Retorna dict com: K, dist, mean_error_px, per_view_error_px (lista),
        rvecs, tvecs (poses do tabuleiro em cada vista, úteis para diagnóstico).
        """
        if self.n_views() < 4:
            raise ValueError("Poucas vistas para calibrar (mínimo recomendado: 4-6).")

        ret, K, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
            self.all_corners, self.all_ids, self.board, self.image_size, None, None
        )

        per_view_errors = []
        obj_points_full = self.board.getChessboardCorners()
        for corners, ids, rvec, tvec in zip(self.all_corners, self.all_ids, rvecs, tvecs):
            obj_pts = obj_points_full[ids.flatten()]
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
            err = np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1)))
            per_view_errors.append(float(err))

        return {
            "K": K,
            "dist": dist,
            "mean_error_px": float(ret),
            "per_view_error_px": per_view_errors,
            "rvecs": rvecs,
            "tvecs": tvecs,
            "image_size": self.image_size,
        }


def solve_extrinsics_from_charuco(charuco_corners, charuco_ids, board, K, dist):
    """
    PnP de uma única detecção contra o tabuleiro de referência.
    Retorna rvec, tvec (pose do tabuleiro em relação à câmera) e o erro
    de reprojeção em pixels dessa mesma detecção.
    """
    obj_points_full = board.getChessboardCorners()
    obj_pts = obj_points_full[charuco_ids.flatten()]
    img_pts = charuco_corners.reshape(-1, 2)

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP falhou para esta detecção.")

    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    err = np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img_pts) ** 2, axis=1)))
    return rvec, tvec, float(err)


def camera_pose_in_board_frame(rvec, tvec):
    """
    solvePnP retorna a pose do TABULEIRO em relação à CÂMERA.
    Esta função inverte para obter a pose da CÂMERA em relação ao TABULEIRO
    (que é o referencial do mundo adotado aqui).
    Retorna (R_cam_in_world, t_cam_in_world) -- t em metros, mesma unidade
    de square_length.
    """
    R_board_in_cam, _ = cv2.Rodrigues(rvec)
    t_board_in_cam = tvec.reshape(3, 1)

    R_cam_in_board = R_board_in_cam.T
    t_cam_in_board = -R_cam_in_board @ t_board_in_cam

    return R_cam_in_board, t_cam_in_board.flatten()
