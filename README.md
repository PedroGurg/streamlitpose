# Calibração multi-câmera (ChArUco) para Pose2Sim

## Changelog desta revisão

**Novo — as imagens de calibração são salvas em disco.** Todo frame usado
na calibração vira PNG, para você poder recalibrar depois sem remontar o
rig nem repetir capturas. A pasta é configurável na sidebar
("Pasta de capturas", padrão `capturas/`):

```
capturas/
  intrinsecos/
    cam_01_frontal/view_001.png ...
    cam_02_lateral/view_001.png ...
    cam_03_diagonal/view_001.png ...
  extrinsecos/
    pose_01/{cam_01_frontal,cam_02_lateral,cam_03_diagonal}.png
    pose_02/...
```

Dois botões novos reprocessam esses arquivos **sem câmera conectada**:

- Aba 2, "♻️ Recalibrar intrínsecos a partir de imagens salvas" — relê os
  PNGs da câmera ativa e refaz a calibração. Útil se você mudar parâmetros
  do tabuleiro ou quiser reprocessar depois de desmontar tudo.
- Aba 3, "♻️ Recarregar poses extrínsecas salvas" — relê as pastas
  `pose_XX` e recalcula as poses com os intrínsecos atuais. Poses em que
  nem todas as 3 câmeras detectam o tabuleiro são ignoradas e listadas.

**Configuração final do rig para 720p em retrato** (dimensionada para
funcionar nos dois cenários possíveis de FOV, já que ele ainda será medido):

| Câmera | Azimute | Distância | Altura (centro óptico) | Inclinação |
|---|---|---|---|---|
| 1 — frontal (eixo da TV) | 0° | 1,74 m | 1,05 m | ~3° para baixo |
| 2 — lateral | 90° | 1,74 m | 1,05 m | ~3° para baixo |
| 3 — diagonal (elevada) | 45° | 1,76 m (canto) | 1,75 m | ~24° para baixo |

- Ponto de mira comum: **0,95 m** do chão
- Deslocamento lateral máximo: **±0,35 m** da marca central, ao longo do
  eixo da câmera 1
- Montagem: câmeras **fisicamente deitadas** + rotação de 90° no app

**Novo — testes automatizados sem hardware.** `test_flow_720p.py` roda o
fluxo completo (intrínsecos, extrínsecos, validação contra verdade de
terreno, exportação do TOML e gravação com barreira) contra três câmeras
virtuais na configuração acima: 32/32 verificações passam.
`test_sync_control.py` é o controle negativo do teste de sincronismo —
mede que a barreira reduz a dispersão entre câmeras de ~5,6 ms
(sequencial) para ~1,3 ms.

**A rotação física NÃO é opcional.** O teste mostrou que, com as câmeras
montadas em paisagem e só rotacionando o vídeo em software, o tabuleiro
no chão projeta fora do quadro (y = 814–1006 numa imagem de 720 px de
altura). Girar pixels não muda o campo de visão: o FOV vertical continua
sendo o do eixo curto do sensor (40,2°), não os 66,1° do eixo longo. As
duas rotações são necessárias e independentes — a física para obter o
FOV, a de software para deixar a imagem em pé.

---

## Changelog das revisões anteriores

**Câmera identificada: Orbbec Astra Pro** — RGB a **1280×720 @ 30 fps**
(não 1080p da Pro Plus, nem 640×480 da Astra base). Defaults do app e
presets do gerador ajustados para esse modo.

**FOV horizontal do modo 720p: MEDIR ANTES DE FIXAR O RIG.** A ficha do
produto lista dois FOVs de RGB (H66,1°/V40,2° e H63,1°/V49,4°), nenhum
declarado explicitamente para o modo 720p da Astra Pro. A diferença muda
a janela de distância utilizável:

| FOV_h real | Distância mínima (retrato, corpo em pé) | Excursão lateral a 1,76 m |
|---|---|---|
| 66,1° | 1,61 m | ± 0,40 m |
| 63,1° | 1,71 m | ± 0,36 m |

Com 63,1° a janela viável fica entre 1,71 m e 1,77 m — praticamente um
ponto. **Procedimento de medição (1 minuto):** fixe a câmera, estenda uma
trena horizontalmente a exatamente 1,70 m dela, perpendicular ao eixo
óptico, e anote a largura visível de borda a borda do quadro. Então
FOV_h = 2 · atan(largura / (2 × 1,70)).

**Tabuleiro NÃO muda.** O de 56×40 cm (8 cm/quadrado) foi revalidado a
1280×720 nos dois cenários de FOV: 24/24 cantos nas três câmeras, com
44–49 px por quadrado e reprojeção de 0,32–0,43 px. Se já imprimiu, está
correto.

**Backend: teste `opencv` primeiro.** Na Astra Pro o stream RGB é uma
câmera UVC separada do pipeline de profundidade, e o `pyorbbecsdk2` (SDK
v2) tem foco nos modelos mais novos — pode não enumerar a Astra Pro. Como
este app usa **apenas RGB**, o backend `opencv` provavelmente é o caminho
certo e o mais simples: sem negociação de formato, sem SDK. Se a câmera
aparecer como webcam comum no sistema, use `opencv` e ignore o backend
`orbbec`.

---

## Changelog da revisão anterior

**Corrigido — bug que impedia a aba 5 de gravar.** Em `camera_worker`, o
acesso `st.session_state.backends[name]` rodava numa thread secundária.
Threads secundárias não herdam o `ScriptRunContext` do Streamlit e recebem
um objeto de estado vazio, o que levantava
`AttributeError: st.session_state has no attribute "backends"` antes do
laço, matando as três threads e travando a barreira por 10 s até abortar
com mensagem vazia. Os backends passaram a ser resolvidos na thread
principal e passados por argumento. Adicionada também uma rede de
segurança que aborta as barreiras explicitamente se um worker falhar
antes do laço.

**Corrigido — resolução padrão inválida.** O padrão era 1280x960, que não
é um perfil suportado por estas câmeras. Passou para 1920x1080. Os perfis
de ficha são 1920x1080 (Pro Plus) e 640x480.

**Corrigido — formato de cor fixo em RGB.** Em USB 2.0, 1080p normalmente
só existe em MJPG; o `reshape(h, w, 3)` quebraria. O backend agora negocia
o formato (RGB, BGR, MJPG, YUYV, NV12, I420), guarda qual foi aceito e
decodifica conforme — MJPG via `cv2.imdecode`. A mensagem de erro de
conexão passou a listar também o **formato** de cada perfil disponível.

**Corrigido — tabuleiro pequeno demais para o rig.** O padrão de 4 cm por
quadrado rende ~12-16 px por quadrado a 1,70 m e **falha na câmera 3**
(elevada): zero marcadores detectados em simulação. O padrão passou para
8 cm por quadrado (56 x 40 cm), que rende 24/24 cantos nas três câmeras
com reprojeção ~0,4 px no modo 1080p.

**Novo — presets de tabuleiro por modo de vídeo.** O tamanho necessário
depende do modo: a 640x480, o tabuleiro de 8 cm volta a falhar na câmera 3
(7 de 24 cantos). O gerador agora tem dois presets validados, e produz
PNG (300 DPI) e PDF em tamanho de página pronto para gráfica.

| Preset | Modo de vídeo | Tabuleiro | Página | Cantos (cam 1/2 e cam 3) |
|---|---|---|---|---|
| `720p` (padrão) | 1280x720 16:9 — Astra Pro | 7x5 @ 8 cm = 56x40 cm | A2 paisagem | 24/24 e 24/24 |
| `480p` | 640x480 4:3 | 5x4 @ 12 cm = 60x48 cm | A1 paisagem | 12/12 e 12/12 |

```bash
python generate_charuco_board.py --preset 720p --out charuco_board_720p
python generate_charuco_board.py --preset 480p --out charuco_board_480p
```

**Importante:** o tabuleiro precisa corresponder ao modo em que você vai
calibrar E coletar. Se mudar de modo, gere e imprima o outro tabuleiro.


App Streamlit para as etapas de calibração das 3 câmeras: intrínsecos por
câmera e extrínsecos por captura simultânea, com validação e exportação
para `Calib.toml`.

## Instalação

```bash
pip install -r requirements.txt
```

Para as câmeras Orbbec, instale também o SDK e o pacote Python correspondente
(`pyorbbecsdk`) conforme a documentação da Orbbec. **Teste primeiro com o
backend `opencv` e webcams comuns** — a lógica de detecção e calibração é
idêntica; só a fonte do frame muda.

## Rodar

```bash
streamlit run app.py
```

## Fluxo de uso

1. **Setup**: escolha o backend (`opencv` para testar, `orbbec` em campo).
   Preencha "Largura/Altura nativa do sensor" com a resolução de **paisagem**
   que a câmera realmente suporta (1280×720 na Astra Pro; 640×480 costuma existir também) — para o backend `orbbec` isso precisa bater exatamente com um
   perfil da lista que `get_stream_profile_list` devolve; o app mostra os
   perfis disponíveis na mensagem de erro se o valor não bater. Se a câmera
   estiver montada de lado para enquadrar o corpo inteiro em retrato, ajuste
   "Rotação" para 90° ou 270° (aplicada em software depois da captura — não
   muda o que é pedido ao sensor). Conecte as 3 câmeras e confira o preview.

2. **Intrínsecos**: selecione uma câmera por vez. Mova o tabuleiro cobrindo
   todo o quadro (cantos inclusive) e clique em "Capturar vista atual" a
   cada posição — 12 a 15 vistas é uma boa meta. Rode a calibração e
   confira o erro médio (alvo: < 0,5 px). Repita para as 3 câmeras.

3. **Extrínsecos**: com os 3 intrínsecos prontos, coloque o tabuleiro
   parado no centro da área, visível pelas 3 câmeras, e clique em
   "Capturar instante simultâneo". Repita em 2-3 poses (chão, inclinado,
   elevado a ~1 m) — cada clique registra uma pose. Escolha qual pose usar
   como referência do mundo e fixe.

4. **Validação e exportação**: informe a altura e a distância que você
   mediu com trena para cada câmera; o app compara com o que a calibração
   devolveu. Ajuste a transformação de eixos se necessário (veja abaixo).
   Confira o gráfico 3D das posições relativas contra os azimutes
   esperados (0°, 90°, 45°). Exporte o `Calib.toml`.

## Pontos de atenção — confira antes de usar em produção

- **Convenção de eixos.** O referencial do mundo adotado é o do próprio
  tabuleiro na pose escolhida como referência. Se o tabuleiro estiver
  deitado com a face impressa para cima, o eixo Z dele já aponta para
  cima — mas isso depende da ordem de impressão/detecção e **deve ser
  conferido visualmente** no gráfico 3D da aba de validação antes de
  confiar no resultado. A opção `z_to_y_up` na mesma aba converte para a
  convenção Y-para-cima do OpenSim, caso necessário.

- **Formato do `Calib.toml`.** As chaves geradas (`matrix`, `distortions`,
  `rotation`, `translation`, `size`, `fisheye`) seguem o padrão documentado
  do Pose2Sim/AniPose no momento desta implementação. **Compare com um
  `Calib.toml` gerado pelo próprio `Pose2Sim.calibration()`** (rode no
  projeto Demo) antes de importar seus dados em produção, porque o formato
  pode variar entre versões do pacote.

- **Alvo de erro extrínseco em pixels vs. milímetros.** O alvo de < 1 cm do
  Pose2Sim é em unidades do mundo real, não em pixels de reprojeção. O app
  reporta o erro de reprojeção em pixels como um proxy rápido, mas a
  comparação que efetivamente importa é a da trena (altura e distância
  medidas vs. calculadas), na mesma aba.

- **Backend Orbbec.** A API usada foi conferida contra `pyorbbecsdk2`
  2.1.2 (que instala o módulo importável `pyorbbecsdk`): `Context`,
  `query_devices`, `get_device_by_index`, `Pipeline`,
  `get_stream_profile_list`, `get_video_stream_profile`, `get_color_frame`
  e `get_data/get_width/get_height` existem com as assinaturas usadas. Se
  a versão instalada divergir, o erro apontará o método.

- **Tamanho do tabuleiro é função da distância e do modo.** Os presets
  foram dimensionados para câmeras a ~1,70 m. Se aproximar ou afastar o
  rig, rode o gerador de novo e confira o relatório de "tamanho aparente
  de um quadrado" — a meta é 30 px ou mais na distância de trabalho.

- **Interferência de luz estruturada.** Este app usa apenas o stream RGB —
  não é afetado pela possível interferência entre os projetores de
  infravermelho das 3 câmeras. Teste essa interferência separadamente, se
  for usar o canal de profundidade em algum outro momento do projeto.
