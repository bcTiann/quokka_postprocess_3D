from despotic import cloud
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from yt.units import K, mp, kb, mh, planck_constant, cm, m, s, g, erg
from despotic.chemistry import NL99, NL99_GC, GOW
from despotic import cloud
from tqdm import tqdm 
import time
from quokka2s.pipeline.prep import config as cfg
# import yt


cell = cloud()

cell.Tg = 10.0
cell.nH = 0.0031622776601683794
cell.colDen = 5623413251903491.0


co_line_map = []

# --- 設定所有常數 ---

cell.sigmaNT = 2.0e5
cell.comp.xoH2 = 0.1
cell.comp.xpH2 = 0.4
cell.comp.xHe = 0.1


cell.dust.alphaGD   = 3.2e-34    # Dust-gas coupling coefficient
cell.dust.sigma10   = 2.0e-25    # Cross section for 10K thermal radiation
cell.dust.sigmaPE   = 1.0e-21    # Cross section for photoelectric heating
cell.dust.sigmaISRF = 3.0e-22    # Cross section to the ISRF
cell.dust.beta      = 2.0        # Dust spectral index
cell.dust.Zd        = 1.0        # Abundance relative to Milky Way
cell.Td             = 10         # Dust temperature
cell.rad.TCMB       = 2.73       # CMB temperature
cell.rad.TradDust   = 0.0        # IR radiation field seen by the dust
cell.rad.ionRate    = 2.0e-17    # Primary ionization rate
cell.rad.chi        = 1.0        # ISRF normalized to Solar neighborhood

start_time = time.time()  # 2. 記錄開始時間

cell.comp.computeDerived(cell.nH)


# --- 執行核心計算 ---
print("-------set Temp Eq --------")
# cell.setTempEq()
print(f"mu = {cell.comp.mu}")
print(f"Tg = {cell.Tg}")
print(f"Td = {cell.Td}")
print("----------------------------")

attempt_start = time.time()
converged = cell.setChemEq(
    network=NL99_GC,
    evolveTemp="iterateDust",
    tol=1e-6,
    maxTime=1e22,
    maxTempIter=50,
)

duration = time.time() - attempt_start
final_Tg = cell.Tg

print("+++++++++++++++++++++++++++++++++++++++++")
print(f"final={final_Tg:.3g} K | converged={converged} | time={duration:.2f}s")
print("+++++++++++++++++++++++++++++++++++++++++")

# --- 處理計算結果 ---
print(f"abundances final: {cell.chemnetwork.abundances}")
print("\n")
print(f"cell chemabundances: {cell.chemabundances}")

print(f"cell emitters: {cell.emitters}")


# Add CO emitter
print("Adding CO emitter...")
cell.addEmitter("CO", cell.chemabundances["CO"])
cell.addEmitter("C+", cell.chemabundances["C+"])
cell.addEmitter("C", cell.chemabundances["C"])
cell.addEmitter("HCO+", cell.chemabundances["HCO+"])
cell.addEmitter("O", cell.chemabundances["O"])

print(cell.emitters)             # => {'CO': <despotic.emitter.emitter object at 0x...>}
print(f"abundance CO: {cell.emitters['CO'].abundance}")  # => 0.0001999439...
print(f"abundance C+: {cell.emitters['C+'].abundance}")
print(f"abundance C: {cell.emitters['C'].abundance}")
print(f"abundance HCO+: {cell.emitters['HCO+'].abundance}")
print(f"abundance O: {cell.emitters['O'].abundance}")
print(f"abundance e-: {cell.chemabundances['e-']} ")

print("\n")
print("Calculating line luminosities...")
lines = cell.lineLum("CO")[0]["lumPerH"]
print(f"CO lumPerH = {lines}")

lines = cell.lineLum("O")[0]["lumPerH"]
print(f"O lumPerH = {lines}")

lines = cell.lineLum("HCO+")[0]["lumPerH"]
print(f"HCO+ lumPerH = {lines}")

# lines = cell.lineLum("H")[0]["lumPerH"]
# print(f"H lumPerH = {lines}")

lines = cell.lineLum("C")[0]["lumPerH"]
print(f"C lumPerH = {lines}")
# co_int_TB = lines[0]["intTB"]
# co_line_map.append(co_int_TB)


# print(f"co_line_map = {co_line_map}")
print(f"Tg final = {cell.Tg}")
# lines = cell.lineLum("C+")
# cp_int_TB = lines[0]["intTB"]
# print(f"C+_line_map = {cp_int_TB}")

