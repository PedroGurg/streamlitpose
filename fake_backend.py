"""
Backend de câmera sintético para testar o fluxo do app sem hardware.

Renderiza, a 1280x720 (modo RGB da Astra Pro), o que cada uma das três
câmeras virtuais do rig veria, e passa o frame pela MESMA rotação para
retrato que o backend real aplica. Assim o teste exercita o caminho
completo: render -> rotação -> detecção -> calibração -> exportação.

As três câmeras leem o MESMO objeto `world`, então uma captura é
genuinamente simultânea -- é assim que o teste de sincronismo consegue
distinguir barreira funcionando de barreira quebrada.
"""
from __future__ import annotations
import cv2
import numpy as np

from camera_backend import CameraBackend


def look_at(cam_pos, target, up=(0, 0, 1.0), roll_deg=0.0):
    """
    Retorna R, t no formato world->camera (convenção OpenCV).

    `roll_deg` gira a câmera em torno do próprio eixo óptico. Isso é o que
    modela a MONTAGEM FÍSICA de lado: com roll de 90 graus, o eixo longo do
    sensor (1280 px, FOV de 66,1 graus) fica VERTICAL no mundo.

    Atenção: rotacionar a imagem em software NÃO faz isso. Software só
    remapeia linhas e colunas; o campo de visão vertical continua sendo o
    do eixo curto do sensor (720 px, 40,2 graus). As duas rotações são
    necessárias e independentes: a física para obter o FOV, a de software
    para deixar a imagem em pé.
    """
    cam_pos = np.asarray(cam_pos, float)
    target = np.asarray(target, float)
    fwd = target - cam_pos
    fwd = fwd / np.linalg.norm(fwd)
    up = np.asarray(up, float)
    if abs(float(fwd @ up)) > 0.99:
        up = np.array([0, 1.0, 0])
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    if roll_deg:
        a = np.deg2rad(roll_deg)
        Rz = np.array([[np.cos(a), -np.sin(a), 0],
                        [np.sin(a), np.cos(a), 0],
                        [0, 0, 1.0]])
        R = Rz @ R
    t = -R @ cam_pos
    return R, t


def intrinsics_landscape(W, H, fov_h_deg):
    """K de uma câmera pinhole em paisagem, com centro óptico ligeiramente
    descentrado (mais realista que cx=W/2 exato)."""
    fx = (W / 2) / np.tan(np.deg2rad(fov_h_deg) / 2)
    return np.array([[fx, 0, W / 2 - 6.0],
                      [0, fx, H / 2 + 4.0],
                      [0, 0, 1]], float)


def expected_K_after_rotation(K, W, H, rotate_deg):
    """
    K equivalente após rotacionar a imagem. Para ROTATE_90_CLOCKWISE, um
    ponto (u,v) da paisagem vai para (H-1-v, u), logo:
        fx' = fy,  fy' = fx,  cx' = (H-1) - cy,  cy' = cx
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if rotate_deg == 0:
        return K.copy()
    if rotate_deg == 90:
        return np.array([[fy, 0, (H - 1) - cy], [0, fx, cx], [0, 0, 1]], float)
    if rotate_deg == 270:
        return np.array([[fy, 0, cy], [0, fx, (W - 1) - cx], [0, 0, 1]], float)
    if rotate_deg == 180:
        return np.array([[fx, 0, (W - 1) - cx], [0, fy, (H - 1) - cy], [0, 0, 1]], float)
    raise ValueError(rotate_deg)


class World:
    """Estado compartilhado pelas três câmeras virtuais."""

    def __init__(self):
        self.mode = "board"          # 'board' | 'person'
        self.board_img = None        # imagem do tabuleiro (grayscale)
        self.board_size_m = (0.0, 0.0)
        self.R_bw = np.eye(3)        # board -> world
        self.t_bw = np.zeros(3)
        self.person_pos = np.array([0.0, 0.0, 0.0])
        self.person_height = 1.75
        self.frame_counter = 0


def _draw_barcode(img, value, n_bits=8):
    """
    Desenha o contador como barras verticais preto/branco no topo da
    imagem. Sobrevive à compressão do vídeo e permite verificar, depois,
    se as três câmeras gravaram o MESMO instante em cada índice de frame.
    """
    h, w = img.shape[:2]
    bar_w = w // n_bits
    bar_h = max(24, h // 12)
    for b in range(n_bits):
        bit = (value >> b) & 1
        color = (255, 255, 255) if bit else (0, 0, 0)
        x0 = b * bar_w
        cv2.rectangle(img, (x0, 0), (x0 + bar_w - 1, bar_h), color, -1)
    return img


def read_barcode(img, n_bits=8):
    h, w = img.shape[:2]
    bar_w = w // n_bits
    bar_h = max(24, h // 12)
    value = 0
    for b in range(n_bits):
        x0 = b * bar_w
        patch = img[2:bar_h - 2, x0 + bar_w // 4: x0 + 3 * bar_w // 4]
        if patch.mean() > 127:
            value |= (1 << b)
    return value


class FakeCameraBackend(CameraBackend):
    def __init__(self, name, K, R_w2c, t_w2c, W, H, world: World, rotate_deg=90):
        self.name = name
        self.K = K
        self.R = R_w2c
        self.t = t_w2c
        self.W = W
        self.H = H
        self.world = world
        self.rotate_deg = rotate_deg
        self.n_reads = 0

    def connect(self):
        pass

    def release(self):
        pass

    def _project(self, pts_world):
        rvec, _ = cv2.Rodrigues(self.R)
        proj, _ = cv2.projectPoints(np.asarray(pts_world, float), rvec, self.t,
                                     self.K, np.zeros(5))
        return proj.reshape(-1, 2)

    def _in_front(self, pts_world):
        cam = (self.R @ np.asarray(pts_world, float).T).T + self.t
        return np.all(cam[:, 2] > 0.05)

    def _render_board(self):
        w = self.world
        bw, bh = w.board_size_m
        # cantos do tabuleiro em coords do tabuleiro; Y cresce para CIMA na
        # imagem gerada pelo OpenCV (verificado empiricamente)
        corners_board = np.array([[0, bh, 0], [bw, bh, 0], [bw, 0, 0], [0, 0, 0]], float)
        corners_world = (w.R_bw @ corners_board.T).T + w.t_bw
        scene = np.full((self.H, self.W, 3), 235, np.uint8)
        if not self._in_front(corners_world):
            return scene
        p = self._project(corners_world).astype(np.float32)
        if not np.all(np.isfinite(p)):
            return scene
        bimg = cv2.cvtColor(w.board_img, cv2.COLOR_GRAY2BGR)
        sh, sw = bimg.shape[:2]
        src = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float32)
        Hm = cv2.getPerspectiveTransform(src, p)
        cv2.warpPerspective(bimg, Hm, (self.W, self.H), dst=scene,
                             flags=cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)
        return scene

    def _render_person(self):
        w = self.world
        scene = np.full((self.H, self.W, 3), 235, np.uint8)
        x, y, z0 = w.person_pos
        hgt = w.person_height
        # tronco como quadrilátero vertical de frente para o centro
        half = 0.18
        ang = np.arctan2(-y, -x)
        dx, dy = -np.sin(ang) * half, np.cos(ang) * half
        torso = np.array([
            [x - dx, y - dy, z0 + 0.35 * hgt],
            [x + dx, y + dy, z0 + 0.35 * hgt],
            [x + dx, y + dy, z0 + 0.85 * hgt],
            [x - dx, y - dy, z0 + 0.85 * hgt],
        ], float)
        head = np.array([[x, y, z0 + 0.95 * hgt]], float)
        if self._in_front(torso) and self._in_front(head):
            pt = self._project(torso).astype(np.int32)
            cv2.fillConvexPoly(scene, pt, (60, 90, 160))
            ph = self._project(head)[0]
            r = max(4, int(0.10 * self.K[0, 0] / max(0.3, np.linalg.norm([x, y, z0 + hgt]))))
            cv2.circle(scene, (int(ph[0]), int(ph[1])), r, (40, 60, 120), -1)
        _draw_barcode(scene, w.frame_counter)
        return scene

    def read(self):
        self.n_reads += 1
        if self.world.mode == "board":
            frame = self._render_board()
        else:
            frame = self._render_person()
        return self._rotate(frame, self.rotate_deg)
