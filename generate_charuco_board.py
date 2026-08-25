"""
Gera o tabuleiro ChArUco para impressão, usando os mesmos parâmetros do
CharucoConfig em charuco_calib.py.

Saídas:
  - charuco_board.png  : imagem do tabuleiro a 300 DPI (tamanho físico exato)
  - charuco_board.pdf  : página A2 paisagem com o tabuleiro centralizado,
                          pronta para enviar à gráfica

Uso:
    python generate_charuco_board.py

O script confere sozinho, ao final, se o tabuleiro gerado é detectável
pelo mesmo código de detecção usado pelo app, e reporta o tamanho aparente
de cada quadrado nas distâncias de trabalho do rig.
"""
import argparse

import numpy as np
import cv2
from PIL import Image

from charuco_calib import CharucoConfig, create_board, detect_charuco

DPI = 300
MM_PER_IN = 25.4

# Presets validados por simulação nas posições reais do rig (1,70 m e 1,76 m).
# O tabuleiro precisa ser dimensionado para o MODO DE VÍDEO usado na coleta:
# a 640x480 o mesmo tabuleiro que funciona em 1080p NÃO é detectado pela
# câmera 3 (elevada) -- 7 de 24 cantos na simulação.
PRESETS = {
    # modo 1280x720 16:9 (Astra Pro) -> 24/24 cantos nas três câmeras,
    # reproj ~0,3-0,4 px, 28-35 px por quadrado. Validado nos dois cenários
    # possíveis de FOV horizontal (63,1 e 66,1 graus).
    "720p": {
        "cfg": dict(squares_x=7, squares_y=5, square_length=0.08, marker_length=0.06),
        "page": (594.0, 420.0),   # A2 paisagem
        "page_name": "A2 paisagem",
    },
    # modo 640x480 4:3 -> 12/12 cantos nas três câmeras
    "480p": {
        "cfg": dict(squares_x=5, squares_y=4, square_length=0.12, marker_length=0.09),
        "page": (841.0, 594.0),   # A1 paisagem
        "page_name": "A1 paisagem",
    },
}

# Definidos em main() conforme o preset escolhido
PAGE_W_MM = 594.0
PAGE_H_MM = 420.0

# Distâncias e FOV do rig, para o relatório de tamanho aparente
RIG = [
    ("cam 1/2 (frontal/lateral)", 1.70),
    ("cam 3 (diagonal elevada)", 1.76),
]
# A Astra Pro entrega RGB a 1280x720. O FOV horizontal exato desse modo nao
# esta estabelecido nas fichas disponiveis -- os dois candidatos sao 63,1 e
# 66,1 graus. O relatorio abaixo mostra os dois; MEÇA o seu (ver README).
FOV_H_DEG = 63.1
FOV_H_DEG_ALT = 66.1


def mm_to_px(mm: float) -> int:
    return round(mm / MM_PER_IN * DPI)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="720p",
                    help="Modo de video da coleta. '720p' (padrao) para "
                         "1280x720 16:9, que e o modo RGB da Astra Pro; "
                         "'480p' para 640x480 4:3.")
    ap.add_argument("--out", default="charuco_board",
                    help="Prefixo dos arquivos de saida.")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    global PAGE_W_MM, PAGE_H_MM
    PAGE_W_MM, PAGE_H_MM = preset["page"]

    cfg = CharucoConfig(**preset["cfg"])
    board, dictionary = create_board(cfg)

    board_w_mm = cfg.squares_x * cfg.square_length * 1000
    board_h_mm = cfg.squares_y * cfg.square_length * 1000

    if board_w_mm > PAGE_W_MM or board_h_mm > PAGE_H_MM:
        raise SystemExit(
            f"Tabuleiro ({board_w_mm:.0f}x{board_h_mm:.0f} mm) nao cabe em A2 "
            f"paisagem ({PAGE_W_MM:.0f}x{PAGE_H_MM:.0f} mm). Reduza "
            "square_length ou o numero de quadrados."
        )

    # --- PNG: so o tabuleiro, em tamanho fisico exato -----------------------
    out_w, out_h = mm_to_px(board_w_mm), mm_to_px(board_h_mm)
    board_img = board.generateImage((out_w, out_h), marginSize=0, borderBits=1)
    cv2.imwrite(f"{args.out}.png", board_img)

    # --- PDF: pagina A2 paisagem com o tabuleiro centralizado ---------------
    page_w_px, page_h_px = mm_to_px(PAGE_W_MM), mm_to_px(PAGE_H_MM)
    page = np.full((page_h_px, page_w_px), 255, np.uint8)
    x0 = (page_w_px - out_w) // 2
    y0 = (page_h_px - out_h) // 2
    page[y0:y0 + out_h, x0:x0 + out_w] = board_img
    Image.fromarray(page).save(f"{args.out}.pdf", "PDF", resolution=DPI)

    margin_w_mm = (PAGE_W_MM - board_w_mm) / 2
    margin_h_mm = (PAGE_H_MM - board_h_mm) / 2

    # --- Autoverificacao: o tabuleiro gerado e detectavel? ------------------
    check = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    c, ids, mc, mids = detect_charuco(check, board, dictionary)
    n_markers = 0 if mids is None else len(mids)
    n_corners = 0 if ids is None else len(ids)
    expected_corners = (cfg.squares_x - 1) * (cfg.squares_y - 1)

    print("=" * 64)
    print("TABULEIRO GERADO")
    print("=" * 64)
    print(f"  Preset        : {args.preset}  (modo de video da coleta)")
    print(f"  Arquivos      : {args.out}.png ({out_w}x{out_h}px @ {DPI} DPI)")
    print(f"                  {args.out}.pdf ({preset["page_name"]}, pronto p/ grafica)")
    print(f"  Tabuleiro     : {cfg.squares_x}x{cfg.squares_y} quadrados")
    print(f"  Tamanho       : {board_w_mm/10:.1f} x {board_h_mm/10:.1f} cm")
    print(f"  Quadrado      : {cfg.square_length*100:.1f} cm")
    print(f"  Marcador      : {cfg.marker_length*100:.1f} cm")
    print(f"  Dicionario    : {cfg.dict_name}")
    print(f"  Padrao legado : {cfg.legacy_pattern}  <-- deixe o checkbox do app IGUAL a isto")
    print(f"  Margem no PDF : {margin_w_mm:.0f} mm laterais, {margin_h_mm:.0f} mm topo/base")
    print()
    print("VERIFICACAO DE DETECCAO")
    print(f"  Marcadores detectados : {n_markers}")
    ok = "OK" if n_corners == expected_corners else "*** FALHA ***"
    print(f"  Cantos ChArUco        : {n_corners} de {expected_corners} esperados  {ok}")
    print()
    print("TAMANHO APARENTE DE UM QUADRADO NO RIG (px)")
    for W, fov, modo in [(1280, FOV_H_DEG, "1280x720 16:9  (FOV_h 63.1)"),
                          (1280, FOV_H_DEG_ALT, "1280x720 16:9  (FOV_h 66.1)"),
                          (640, FOV_H_DEG, "640x480   4:3  (FOV_h 63.1)")]:
        fx = (W / 2) / np.tan(np.deg2rad(fov) / 2)
        print(f"  modo {modo}:")
        for lbl, d in RIG:
            px = cfg.square_length * fx / d
            flag = "ok" if px >= 30 else "BAIXO (mire >= 30 px)"
            print(f"      {lbl:28s} {px:5.1f} px  {flag}")
    print()
    print("IMPRESSAO")
    print("  - Imprima em 'tamanho real / 100%'. NUNCA 'ajustar a pagina'.")
    print(f"  - Confira com regua: um quadrado deve medir {cfg.square_length*100:.1f} cm.")
    print("  - Cole em superficie RIGIDA e plana (foam board / PVC). Papel ondulado")
    print("    introduz erro que a calibracao vai atribuir a lente.")
    print("  - Fosco, nao brilhante: reflexo apaga marcador sob luz direta.")
    print("=" * 64)


if __name__ == "__main__":
    main()
