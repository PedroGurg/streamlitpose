"""
Controle negativo do teste de sincronismo.

Um teste que passa sempre não prova nada. Aqui um "relógio" avança em
segundo plano DURANTE as leituras, de modo que ler as câmeras em
instantes diferentes produz valores diferentes. Compara-se então:

  A) leitura SEQUENCIAL (uma câmera após a outra) -- o que o código fazia
     antes da barreira
  B) leitura com BARREIRA (as 3 threads liberadas juntas)

Se a barreira funciona, B deve ter dispersão muito menor que A.
"""
import threading
import time

import numpy as np

from fake_backend import World, FakeCameraBackend, look_at, intrinsics_landscape

W_NAT, H_NAT, FOV_H, ROTATE, ROLL = 1280, 720, 66.1, 90, -90
AIM = np.array([0.0, 0.0, 0.95])
RIG = {
    "cam_01_frontal": np.array([1.70, 0.00, 1.05]),
    "cam_02_lateral": np.array([0.00, 1.70, 1.05]),
    "cam_03_diagonal": np.array([1.24, 1.24, 1.75]),
}
CAM_NAMES = list(RIG)

world = World()
world.mode = "person"
K = intrinsics_landscape(W_NAT, H_NAT, FOV_H)
backends = {}
for name, pos in RIG.items():
    R, t = look_at(pos, AIM, roll_deg=ROLL)
    backends[name] = FakeCameraBackend(name, K, R, t, W_NAT, H_NAT, world, ROTATE)

# Relógio de alta resolução: avança a cada 1 ms, como o tempo real avança
# entre as leituras de câmeras diferentes.
tick_stop = threading.Event()
tick_val = {"t": 0}


def ticker():
    while not tick_stop.is_set():
        tick_val["t"] += 1
        world.frame_counter = tick_val["t"] % 256
        time.sleep(0.001)


N = 40


def measure_sequential():
    spreads = []
    for _ in range(N):
        vals = []
        for name in CAM_NAMES:
            backends[name].read()
            vals.append(world.frame_counter)
        spreads.append(max(vals) - min(vals))
        time.sleep(0.005)
    return spreads


def measure_barrier():
    spreads = []
    n = len(CAM_NAMES)
    bg = threading.Barrier(n + 1)
    bd = threading.Barrier(n + 1)
    captured = {}
    stop = threading.Event()

    def worker(name, backend):
        while not stop.is_set():
            try:
                bg.wait(timeout=5)
            except threading.BrokenBarrierError:
                return
            backend.read()
            captured[name] = world.frame_counter
            try:
                bd.wait(timeout=5)
            except threading.BrokenBarrierError:
                return

    ths = [threading.Thread(target=worker, args=(n_, backends[n_]), daemon=True)
           for n_ in CAM_NAMES]
    for t in ths:
        t.start()
    for _ in range(N):
        bg.wait(timeout=5)
        bd.wait(timeout=5)
        vals = [captured[n_] for n_ in CAM_NAMES]
        spreads.append(max(vals) - min(vals))
        time.sleep(0.005)
    stop.set()
    bg.abort()
    bd.abort()
    for t in ths:
        t.join(timeout=1)
    return spreads


tk = threading.Thread(target=ticker, daemon=True)
tk.start()
time.sleep(0.05)

seq = measure_sequential()
bar = measure_barrier()

tick_stop.set()
tk.join(timeout=1)

print("=" * 70)
print("CONTROLE NEGATIVO DO TESTE DE SINCRONISMO")
print("=" * 70)
print(f"  amostras por modo: {N}   (relógio avança 1 unidade por ms)")
print()
for lbl, data in [("SEQUENCIAL (sem barreira)", seq), ("BARREIRA", bar)]:
    a = np.array(data, float)
    print(f"  {lbl:26s} dispersão média={a.mean():6.2f} ms   "
          f"máx={a.max():5.0f} ms   zero em {(a == 0).mean()*100:5.1f}% dos quadros")
print()
ok = np.mean(bar) < np.mean(seq)
print(f"  Barreira reduz a dispersão: {'SIM' if ok else 'NÃO'}")
print()
print("  Nota: mesmo com barreira a dispersão não é zero -- ela sincroniza a")
print("  CHAMADA de leitura, não a exposição do sensor. Com hardware real e")
print("  sem trigger, some a isso o buffer do driver e o resíduo de meio")
print("  quadro a 30 fps (±16,7 ms).")
print("=" * 70)
