"""
Abstração de fonte de vídeo para o app de calibração.

Dois backends:
- OpenCVCameraBackend: qualquer câmera enxergada como dispositivo USB genérico
  pelo OpenCV (cv2.VideoCapture). Funciona para testar o app com webcams
  antes de ir a campo com as Orbbec.
- OrbbecCameraBackend: usa o pyorbbecsdk para ler o stream de cor das
  câmeras Orbbec. A API exata (nomes de classes/métodos) pode variar entre
  versões do SDK -- confira a documentação instalada localmente
  (import pyorbbecsdk; help(pyorbbecsdk)) e ajuste se necessário.
"""

from __future__ import annotations
import cv2
import numpy as np


_ROTATE_CODES = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class CameraBackend:
    """Interface comum. Qualquer backend deve implementar estes métodos."""

    def connect(self) -> None:
        raise NotImplementedError

    def read(self) -> np.ndarray | None:
        """Retorna um frame BGR (np.ndarray) ou None se falhar."""
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    @staticmethod
    def _rotate(frame: np.ndarray | None, rotate_deg: int) -> np.ndarray | None:
        """
        Aplica rotação em software ao frame. Necessário quando o sensor é
        físico 4:3 (ou qualquer paisagem nativa) montado de lado para
        capturar o corpo inteiro em retrato -- a câmera entrega os pixels
        como o sensor os viu, sem saber da orientação física do rig.
        """
        if frame is None or rotate_deg == 0:
            return frame
        code = _ROTATE_CODES.get(rotate_deg)
        if code is None:
            raise ValueError(f"Rotação inválida: {rotate_deg} (use 0, 90, 180 ou 270)")
        return cv2.rotate(frame, code)


class OpenCVCameraBackend(CameraBackend):
    def __init__(self, index: int, width: int = 1280, height: int = 720, rotate_deg: int = 0):
        self.index = index
        self.width = width
        self.height = height
        self.rotate_deg = rotate_deg
        self.cap: cv2.VideoCapture | None = None

    def connect(self) -> None:
        self.cap = cv2.VideoCapture(self.index)
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera de índice {self.index}")

    def read(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        frame = frame if ok else None
        return self._rotate(frame, self.rotate_deg)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class OrbbecCameraBackend(CameraBackend):
    """
    Backend para câmeras Orbbec via pyorbbecsdk.

    ATENÇÃO: a estrutura abaixo segue o padrão típico do SDK (Pipeline /
    Config / start / wait_for_frames / color_frame), mas nomes exatos de
    classes e métodos podem diferir conforme a versão instalada. Rode
    primeiro com o backend OpenCV para validar o app; troque para este
    backend em campo e ajuste conforme o erro apontar.
    """

    def __init__(self, device_index: int, width: int = 1280, height: int = 720,
                 fps: int = 30, rotate_deg: int = 0):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.rotate_deg = rotate_deg
        self.pipeline = None
        self.color_format = None   # definido na conexão, conforme perfil negociado

    def connect(self) -> None:
        try:
            from pyorbbecsdk import Context, Pipeline, Config, OBSensorType, OBFormat
        except ImportError as e:
            raise RuntimeError(
                "pyorbbecsdk não encontrado. Instale o pacote 'pyorbbecsdk2' "
                "(que fornece o módulo importável 'pyorbbecsdk') antes de usar "
                "este backend."
            ) from e

        device_list = Context().query_devices()
        count = device_list.get_count()
        if count == 0:
            raise RuntimeError("Nenhuma câmera Orbbec encontrada.")
        if self.device_index >= count:
            raise RuntimeError(
                f"Índice {self.device_index} inválido: apenas {count} "
                f"câmera(s) Orbbec conectada(s) (índices válidos: 0 a {count - 1})."
            )
        device = device_list.get_device_by_index(self.device_index)
        self.pipeline = Pipeline(device)
        config = Config()
        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)

        # Em USB 2.0, 1080p normalmente só existe em MJPG -- RGB cru não cabe
        # na banda. Tenta os formatos em ordem de preferência (menos
        # processamento primeiro) em vez de exigir RGB.
        preferred = [OBFormat.RGB, OBFormat.BGR, OBFormat.MJPG,
                     OBFormat.YUYV, OBFormat.NV12, OBFormat.I420]
        color_profile = None
        for fmt in preferred:
            try:
                color_profile = profile_list.get_video_stream_profile(
                    self.width, self.height, fmt, self.fps
                )
                break
            except Exception:
                continue

        if color_profile is None:
            available = []
            for i in range(profile_list.get_count()):
                try:
                    p = profile_list.get_stream_profile_by_index(i).as_video_stream_profile()
                    available.append(
                        f"{p.get_width()}x{p.get_height()}@{p.get_fps()}fps "
                        f"[{p.get_format()}]"
                    )
                except Exception:
                    continue
            hint = f" Perfis disponíveis: {available}" if available else ""
            raise RuntimeError(
                f"Nenhum stream de cor {self.width}x{self.height}@{self.fps}fps "
                f"suportado por este dispositivo em nenhum formato conhecido.{hint} "
                "Use a resolução NATIVA do sensor (paisagem) aqui e ajuste o "
                "campo 'Rotação' na sidebar para obter o enquadramento em retrato."
            )

        self.color_format = color_profile.get_format()
        config.enable_stream(color_profile)
        self.pipeline.start(config)

    def read(self) -> np.ndarray | None:
        if self.pipeline is None:
            return None
        frames = self.pipeline.wait_for_frames(1000)
        if frames is None:
            return None
        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None
        frame = self._decode_to_bgr(color_frame)
        return self._rotate(frame, self.rotate_deg)

    def _decode_to_bgr(self, color_frame) -> np.ndarray | None:
        """
        Converte o frame de cor para BGR (convenção OpenCV), conforme o
        formato negociado na conexão. Formatos comprimidos (MJPG) passam
        por cv2.imdecode; formatos crus passam por reshape + cvtColor.
        """
        from pyorbbecsdk import OBFormat

        data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        h, w = color_frame.get_height(), color_frame.get_width()
        fmt = self.color_format

        if fmt == OBFormat.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if fmt == OBFormat.RGB:
            return cv2.cvtColor(data.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
        if fmt == OBFormat.BGR:
            return data.reshape((h, w, 3))
        if fmt == OBFormat.YUYV:
            return cv2.cvtColor(data.reshape((h, w, 2)), cv2.COLOR_YUV2BGR_YUYV)
        if fmt == OBFormat.NV12:
            return cv2.cvtColor(data.reshape((h * 3 // 2, w)), cv2.COLOR_YUV2BGR_NV12)
        if fmt == OBFormat.I420:
            return cv2.cvtColor(data.reshape((h * 3 // 2, w)), cv2.COLOR_YUV2BGR_I420)

        raise RuntimeError(
            f"Formato de cor não tratado: {fmt}. Adicione a conversão "
            "correspondente em OrbbecCameraBackend._decode_to_bgr."
        )

    def release(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None


def make_backend(kind: str, index: int, width: int, height: int, rotate_deg: int = 0) -> CameraBackend:
    if kind == "opencv":
        return OpenCVCameraBackend(index, width, height, rotate_deg=rotate_deg)
    elif kind == "orbbec":
        return OrbbecCameraBackend(index, width, height, rotate_deg=rotate_deg)
    raise ValueError(f"Backend desconhecido: {kind}")
