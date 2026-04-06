from despotic import cloud
from despotic.composition import composition
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from yt.units import K, mp, kb, mh, planck_constant, cm, m, s, g, erg
from despotic.chemistry import NL99, NL99_GC, GOW
from despotic import cloud
from tqdm import tqdm 
import time
from ..pipeline.prep import config as cfg
# import yt


cell = cloud()
GAMMA = 5.0/3.0
mh_cgs = mh.in_cgs().value  # g
kB_cgs = kb.in_cgs().value  # erg/K 
cell.nH = 1e-5
cell.colDen = 1000000000000000.0
test_T = 1000.0 * K
test_mu = 1.4
sim_rho = cell.nH * mh / 0.71  # 粗略估算质量密度
sim_rho = sim_rho.in_cgs().value  # 转换为 cgs 单位 (g/cm^3)
sim_internal_E = (sim_rho * kb * test_T) / (test_mu * mh * (GAMMA - 1.0))
sim_internal_E = sim_internal_E.in_cgs().value  # 转换为 cgs 单位 (erg/cm^3)
print(f"Simulated Input: nH={cell.nH:.1e}, E_int={sim_internal_E:.2e} erg/cm^3")

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


attempt_start = time.time()
# converged = cell.setChemEq(
#     network=NL99_GC,
#     evolveTemp="iterateDust",
#     tol=1e-6,
#     maxTime=1e26,
#     maxTempIter=500,
# )
current_mu = 1.4  # 初始猜测值

max_iter = 20
tolerance = 1e-3
for i in range(max_iter):
    
    # --- A. 根据 E_int 和 mu 计算温度 ---
    # 公式: T = (E_int * (gamma-1) * mu * mH) / (rho * kB)
    # 这里的 rho 我们用 Despotic 的定义: nH (H核密度)
    # 注意：Despotic 的 mu 是 "mean mass per particle in units of mH"
    # 理想气体定律变形: P = n_tot * kB * T
    # P = (gamma - 1) * E_int
    # n_tot = (rho) / (mu * mH)  <-- 这是一个很好的近似
    
    # 让我们用最稳健的公式：
    # T = (gamma - 1) * (E_int / rho) * mu * (mH / kB)
    # 其中 E_int/rho 就是 specific internal energy (erg/g)
    
    T_calc = (current_gamma - 1.0) * (sim_internal_E / sim_rho) * current_mu * (mh_cgs / kB_cgs)
    
    # --- B. 安全限制 (Clamping) ---
    # 防止 T 变成 0 或 无穷大，导致 Despotic 报错
    if T_calc < 2.73: T_calc = 2.73
    if T_calc > 1e8: T_calc = 1e8 
    
    # --- C. 赋值给 Cell ---
    cell.Tg = T_calc
    
    # --- D. 运行 Despotic (只算化学，不解热平衡) ---
    # 关键点：evolveTemp='fixed'
    converged = cell.setChemEq(
        network=NL99_GC,
        evolveTemp='fixed', 
        tol=1e-6,
        maxTime=1e10  # 化学平衡很快，不需要很长时间
    )
    
    # --- E. 获取新的 mu ---
    # 获取当前丰度
    abundances = cell.chemnetwork.abundances
    # 计算新的 mu (Despotic 自带函数)
    new_mu = cell.chemnetwork.mu()
    # 用临时 composition 计算 cv，避免修改主 cell.comp 导致其他流程问题
    temp_comp = composition()
    temp_comp.xHI = abundances.get('H', 0.0)
    temp_comp.xHplus = abundances.get('H+', 0.0)
    h2 = abundances.get('H2', 0.0)
    temp_comp.xpH2 = 0.5 * h2
    temp_comp.xoH2 = 0.5 * h2
    temp_comp.xHe = abundances.get('He', 0.0) + abundances.get('He+', 0.0)
    temp_comp.xe = abundances.get('e-', 0.0)
    temp_comp.computeDerived(cell.nH)
    cv = temp_comp.computeCv(cell.Tg)
    current_gamma = (cv + 1.0) / cv if cv > 0.0 else 1.0
    
    print(f"Iter {i+1}: Input T={T_calc:.2f} K | Result mu={new_mu:.4f} | Converged={converged}")
    
    # --- F. 检查收敛 ---
    if abs(new_mu - current_mu) / current_mu < tolerance:
        print(">>> Converged!")
        break
        
    current_mu = new_mu

# ==========================================
# 4. 结果输出
# ==========================================
print("\n------- Final Results --------")
print(f"Final T: {cell.Tg:.2f} K")
print(f"Final mu: {current_mu:.4f}")
print(f"Abundances: C+={cell.chemnetwork.abundances['C+']:.2e}, CO={cell.chemnetwork.abundances['CO']:.2e}")
print("\n------- Final Results ENDS--------")
print("\n------------------------------------------------")



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

Eint_dimless = cell.comp.computeEint(cell.Tg)
Eint_perH_in_erg = Eint_dimless * kb * cell.Tg * K
print(f"Eint_dimless = {Eint_dimless}")
print(f"internal energy per H nucleus in erg = {Eint_perH_in_erg} ")