"""
PI-NFISSDR v2: Physics-Informed ANFIS для дозирования сорбента при очистке птиц
=================================================================================
ПЕРЕМЕННЫЕ (X4 удалена, нумерация сдвинута):
  x1 = BodySize       (700–7000 г)
  x2 = OilCoverage    (0–100 %)
  x3 = FeatherDensity (0–100 %)
  x4 = OilViscosity   (1–40 мм²/с)   ← была x5
  x5 = OilAge         (0–100 у.е.)   ← была x6
  Y  = SorbentDose    (0–800 г)

НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ (см. раздел "HYPERPARAMETERS"):
  N_RULES    — количество правил (5..10, рекомендуется 5)
  N_EPOCHS   — число эпох обучения (200..2000)
  LR         — скорость обучения MF (0.001..0.05)
  LAMBDA_PHY — вес физических ограничений (0.01..1.0)
  LAMBDA_MON — вес штрафа монотонности (0.0..0.5)
  SIGMA_INIT — начальная ширина MF (0.1..0.4)
  NOISE_STD  — шум при генерации данных (0..30)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ A: НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ  ← изменяйте здесь
# ─────────────────────────────────────────────────────────────
N_RULES    = 5       # число нечётких правил (5 = по числу переменных)
N_EPOCHS   = 500     # число эпох обучения
LR         = 0.005    # скорость обучения для центров/ширин MF
LAMBDA_PHY = 0.3     # вес физических ограничений в функции потерь
LAMBDA_MON = 0.1     # вес штрафа за нарушение монотонности
SIGMA_INIT = 0.25    # начальная ширина гауссовых MF (в норм. пространстве [0,1])
N_SAMPLES  = 1500    # число обучающих примеров
NOISE_STD  = 10      # шум генерации данных (г)
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ B: ОПРЕДЕЛЕНИЕ ПЕРЕМЕННЫХ
# ─────────────────────────────────────────────────────────────
VAR_RANGES = {
    'x1_BodySize':       (700,  7000),   # г
    'x2_OilCoverage':    (0,    100),    # %
    'x3_FeatherDensity': (0,    100),    # %
    'x4_OilViscosity':   (1,    40),     # мм²/с
    'x5_OilAge':         (0,    100),    # у.е.
}
Y_RANGE = (0, 800)   # г

VAR_LABELS = {
    'x1_BodySize':       ('x₁', 'BodySize',       'g'),
    'x2_OilCoverage':    ('x₂', 'OilCoverage',    '%'),
    'x3_FeatherDensity': ('x₃', 'FeatherDensity', '%'),
    'x4_OilViscosity':   ('x₄', 'OilViscosity',   'mm²/s'),
    'x5_OilAge':         ('x₅', 'OilAge',         'arb. u.'),
}

COLORS = ['#1a3a6b', '#1a7a6b', '#c0392b', '#f39c12', '#7d3c98']
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ C: ФИЗИЧЕСКИ ОБОСНОВАННЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────
# Параметры физических функций (можно настраивать):
ALPHA = 0.7    # степень нелинейности OilCoverage (0.5..1.0)
BETA  = 1.5    # степень нелинейности FeatherDensity (1.0..2.5)
GAMMA = 3.0    # насыщение OilAge (1.0..5.0)
# Веса физических вкладов (сумма = 1):
W_PHY = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

def physics_predict(X_norm):
    """
    Физически обоснованный прогноз дозы сорбента.
    X_norm: (N,5) — нормализованные входы [0,1]
    Возвращает Y_phys в граммах [0, 800]
    """
    x1n, x2n, x3n, x4n, x5n = X_norm[:,0], X_norm[:,1], X_norm[:,2], X_norm[:,3], X_norm[:,4]
    f1 = x1n                              # линейно по массе тела
    f2 = x2n ** ALPHA                     # нелинейно по покрытию нефтью
    f3 = x3n ** BETA                      # нелинейно по плотности пера
    f4 = np.log1p(x4n * 39) / np.log1p(39)  # лог. по вязкости
    f5 = 1.0 - np.exp(-GAMMA * x5n)      # насыщающая по возрасту загрязнения
    y_phys = Y_RANGE[1] * (W_PHY[0]*f1 + W_PHY[1]*f2 + W_PHY[2]*f3 +
                            W_PHY[3]*f4 + W_PHY[4]*f5)
    return np.clip(y_phys, 0, Y_RANGE[1])

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ D: ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДАННЫХ
# ─────────────────────────────────────────────────────────────
def generate_data(n_samples=N_SAMPLES, noise_std=NOISE_STD, seed=RANDOM_SEED):
    """
    Генерация данных на основе физической модели + экспертных поправок.
    Физически обоснованная зависимость: большая птица + сильное загрязнение
    + густое перо + высокая вязкость + старое загрязнение → больше сорбента.
    """
    np.random.seed(seed)
    keys = list(VAR_RANGES.keys())
    X_raw = np.column_stack([
        np.random.uniform(lo, hi, n_samples)
        for lo, hi in VAR_RANGES.values()
    ])
    # Нормализация
    X_norm = np.column_stack([
        (X_raw[:, i] - VAR_RANGES[k][0]) / (VAR_RANGES[k][1] - VAR_RANGES[k][0])
        for i, k in enumerate(keys)
    ])
    Y = physics_predict(X_norm)
    # Добавить реалистичный шум
    Y += np.random.normal(0, noise_std, n_samples)
    Y = np.clip(Y, 0, Y_RANGE[1])
    df = pd.DataFrame(X_raw, columns=keys)
    df['Y_SorbentDose'] = Y
    return df, X_norm, Y

print("=" * 60)
print("PI-NFISSDR v2: Generating data...")
df, X_norm_all, Y_all = generate_data()
print(f"Created {len(df)} samples.")
print(f"Y: min={Y_all.min():.1f}, mean={Y_all.mean():.1f}, max={Y_all.max():.1f} g")

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ E: НАЧАЛЬНЫЕ ЦЕНТРЫ ПРАВИЛ (5 ПРАВИЛ)
# ─────────────────────────────────────────────────────────────
#
# 5 правил соответствуют 5 "физическим сценариям":
#
# Правило 1 (Минимальное загрязнение):
#   Маленькая птица, малое покрытие, редкое перо, низкая вязкость, свежее → Low
# Правило 2 (Малое загрязнение + вязкость):
#   Маленькая птица, малое покрытие, нормальное перо, средняя вязкость, среднее → Low-Medium
# Правило 3 (Среднее загрязнение):
#   Средняя птица, среднее покрытие, нормальное перо, средняя вязкость, среднее → Medium
# Правило 4 (Сильное загрязнение):
#   Крупная птица, высокое покрытие, густое перо, высокая вязкость, старое → Large
# Правило 5 (Критическое загрязнение):
#   Очень крупная птица, полное покрытие, плотное перо, очень вязкое, старое → VeryLarge
#
# Центры в нормализованном пространстве [0, 1]:
INIT_CENTERS = np.array([
    # x1,   x2,   x3,   x4,   x5
    [0.10,  0.10,  0.10,  0.10,  0.10],   # Правило 1: всё минимально
    [0.25,  0.25,  0.35,  0.35,  0.30],   # Правило 2: умеренно-низкое
    [0.50,  0.50,  0.50,  0.50,  0.50],   # Правило 3: среднее
    [0.75,  0.75,  0.70,  0.70,  0.70],   # Правило 4: высокое
    [0.95,  0.95,  0.90,  0.90,  0.90],   # Правило 5: критическое
], dtype=float)

# Физически обоснованные начальные консеквенты (выходы правил в граммах):
# Получены из physics_predict на центрах правил
INIT_Y_RULES = physics_predict(INIT_CENTERS)
print(f"\nInitial rule outputs (physics-based prediction): {INIT_Y_RULES.round(1)} g")

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ F: КЛАСС PI-ANFIS
# ─────────────────────────────────────────────────────────────
class PIANFIS:
    """
    Physics-Informed ANFIS (Takagi-Sugeno 1-го порядка).

    Структура (5 слоёв):
      Слой 1: Гауссовые функции принадлежности μᵢₖ(xᵢ)
      Слой 2: Степени активации wₖ = ∏ᵢ μᵢₖ(xᵢ)
      Слой 3: Нормализация w̄ₖ = wₖ / Σⱼ wⱼ
      Слой 4: Линейный вывод Такаги–Сугэно: yₖ = p₀ᵏ + Σᵢ pᵢᵏ·x̃ᵢ
      Слой 5: Дефаззификация ŷ = Σₖ w̄ₖ·yₖ

    Функция потерь:
      L = L_data + λ_phy·L_physics + λ_mon·L_monotone

    Обучение: гибридный алгоритм
      - Линейные параметры {pᵢᵏ}: метод наименьших квадратов (LSE)
      - Нелинейные параметры {cᵢₖ, σᵢₖ}: градиентный спуск по полной L
    """
    def __init__(self, n_rules=N_RULES, lr=LR,
                 lambda_phy=LAMBDA_PHY, lambda_mon=LAMBDA_MON,
                 n_epochs=N_EPOCHS, sigma_init=SIGMA_INIT):
        self.n_rules    = n_rules
        self.lr         = lr
        self.lambda_phy = lambda_phy
        self.lambda_mon = lambda_mon
        self.n_epochs   = n_epochs
        self.n_inputs   = 5

        # Нелинейные параметры (обучаются градиентом)
        self.C = INIT_CENTERS.copy()                          # (n_rules, 5) — центры MF
        self.S = np.full((n_rules, 5), sigma_init, dtype=float)  # (n_rules, 5) — ширины MF

        # Линейные параметры (обучаются LSE): p[k, i] для k-го правила
        # Инициализация: физический прогноз на центре правила
        self.P = np.zeros((n_rules, self.n_inputs + 1))  # (n_rules, 6): [p0, p1..p5]
        for k in range(n_rules):
            self.P[k, 0] = INIT_Y_RULES[k]   # константный член = физический прогноз

    # ----------------------------------------------------------
    def _mf(self, X):
        """
        Вычисление гауссовых MF.
        X: (N, 5) → μ: (N, n_rules, 5)
        """
        # X[:, None, :] → (N, 1, 5), C[None,:,:] → (1, R, 5)
        diff = X[:, None, :] - self.C[None, :, :]    # (N, R, 5)
        mu = np.exp(-0.5 * (diff / (self.S[None,:,:] + 1e-8))**2)  # (N, R, 5)
        return mu

    def _forward(self, X):
        """
        Прямой проход.
        X: (N, 5) нормализованные входы
        Возвращает: ŷ (N,), W_bar (N, R), phi (N, R*(n+1))
        """
        N = X.shape[0]
        mu = self._mf(X)                        # (N, R, 5)
        W = mu.prod(axis=2)                     # (N, R) — произведение MF по входам
        W_sum = W.sum(axis=1, keepdims=True) + 1e-10
        W_bar = W / W_sum                       # (N, R) — нормализованные веса

        # Регрессионная матрица для LSE: каждая строка = [w̄ₖ·[1, x̃]] для всех k
        X_aug = np.hstack([np.ones((N, 1)), X])  # (N, 6)
        phi = np.einsum('nr,nf->nrf', W_bar, X_aug).reshape(N, -1)  # (N, R*6)

        y_hat = phi @ self.P.ravel()            # (N,)
        y_hat = np.clip(y_hat, 0, Y_RANGE[1])
        return y_hat, W_bar, phi

    def _lse_update(self, phi, y_true):
        """
        LSE для линейных параметров P.
        phi: (N, R*6), y_true: (N,)
        """
        try:
            p_flat, _, _, _ = np.linalg.lstsq(phi, y_true, rcond=None)
            self.P = p_flat.reshape(self.n_rules, self.n_inputs + 1)
        except Exception:
            pass

    def _physics_loss(self, X, y_hat):
        """
        Физические ограничения:
        L_phys  = MSE(ŷ, ŷ_phys)  — прогноз должен быть близок к физической модели
        L_range = штраф за выход за [0, 800]
        """
        y_phys = physics_predict(X)
        L_phys = np.mean((y_hat - y_phys) ** 2)
        L_range = np.mean(np.maximum(0, y_hat - Y_RANGE[1])**2 +
                          np.maximum(0, -y_hat)**2)
        return L_phys + L_range

    def _monotone_loss(self, X, eps=1e-3):
        """
        Штраф за нарушение монотонности: ∂ŷ/∂xᵢ ≥ 0 для всех i.
        Аппроксимируем конечными разностями.
        """
        total = 0.0
        for i in range(self.n_inputs):
            X_fwd = X.copy(); X_fwd[:, i] += eps
            X_bwd = X.copy(); X_bwd[:, i] -= eps
            y_fwd, _, _ = self._forward(np.clip(X_fwd, 0, 1))
            y_bwd, _, _ = self._forward(np.clip(X_bwd, 0, 1))
            grad = (y_fwd - y_bwd) / (2 * eps)
            total += np.mean(np.maximum(0, -grad) ** 2)  # штраф только за отрицательный градиент
        return total

    def _grad_C_S(self, X, y_true, y_hat, W_bar):
        """
        Градиент полной функции потерь по C и S (аналитически).
        """
        N = X.shape[0]
        residuals = y_hat - y_true   # (N,)

        # Выходы правил: yₖ = p0ₖ + Σ pᵢₖ·xᵢ
        X_aug = np.hstack([np.ones((N, 1)), X])      # (N, 6)
        rule_outs = X_aug @ self.P.T                  # (N, R)

        # ∂ŷ/∂w̄ₖ = yₖ (при нормализованных весах — упрощённо)
        W_sum = W_bar.sum(axis=1, keepdims=True) + 1e-10

        grad_C = np.zeros_like(self.C)
        grad_S = np.zeros_like(self.S)

        mu = self._mf(X)   # (N, R, 5)
        W  = mu.prod(axis=2)    # (N, R)

        for k in range(self.n_rules):
            # ∂ŷ/∂wₖ = (yₖ - ŷ) / W_sum  (частная производная по wₖ при нормализации)
            dydwk = (rule_outs[:, k] - y_hat) / (W_sum.ravel() + 1e-10)   # (N,)
            dL_dwk = residuals * dydwk   # (N,)

            for i in range(self.n_inputs):
                # ∂wₖ/∂μᵢₖ = wₖ / μᵢₖ  (произведение остальных MF)
                mu_ik = mu[:, k, i] + 1e-10
                dw_dmu = W[:, k] / mu_ik   # (N,)

                # ∂μᵢₖ/∂cᵢₖ = μᵢₖ · (xᵢ - cᵢₖ) / σᵢₖ²
                diff = X[:, i] - self.C[k, i]
                sigma2 = self.S[k, i]**2 + 1e-10
                dmu_dc = mu_ik * diff / sigma2     # (N,)
                dmu_ds = mu_ik * diff**2 / (self.S[k, i]**3 + 1e-10)   # (N,)

                chain = dL_dwk * dw_dmu    # (N,)
                grad_C[k, i] = np.mean(chain * dmu_dc)
                grad_S[k, i] = np.mean(chain * dmu_ds)

        return grad_C, grad_S

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose_every=50):
        """Гибридное обучение."""
        history = {'train_rmse': [], 'val_rmse': [], 'L_data': [], 'L_phys': [], 'L_mon': []}
        N = X_train.shape[0]

        for epoch in range(self.n_epochs):
            # --- Прямой проход ---
            y_hat, W_bar, phi = self._forward(X_train)

            # --- LSE для линейных параметров (каждые 5 эпох) ---
            if epoch % 5 == 0:
                self._lse_update(phi, y_train)
                y_hat, W_bar, phi = self._forward(X_train)  # пересчитать после LSE

            # --- Функция потерь ---
            L_data = np.mean((y_hat - y_train)**2)
            L_phys = self._physics_loss(X_train, y_hat)
            L_mon  = self._monotone_loss(X_train) if self.lambda_mon > 0 else 0.0
            L_total = L_data + self.lambda_phy * L_phys + self.lambda_mon * L_mon

            # --- Градиентный шаг по C, S ---
            grad_C, grad_S = self._grad_C_S(X_train, y_train, y_hat, W_bar)

            # Добавить физический градиент (аппроксимация)
            if self.lambda_phy > 0:
                y_phys = physics_predict(X_train)
                phys_res = y_hat - y_phys
                grad_C_p, grad_S_p = self._grad_C_S(X_train, y_phys, y_hat, W_bar)
                grad_C += self.lambda_phy * grad_C_p
                grad_S += self.lambda_phy * grad_S_p

            self.C -= self.lr * np.clip(grad_C, -5, 5)
            self.S -= self.lr * np.clip(grad_S, -5, 5)

            # Ограничения: центры в [0,1], σ > 0.05
            self.C = np.clip(self.C, 0.0, 1.0)
            self.S = np.clip(self.S, 0.05, 1.0)

            # --- История ---
            rmse = np.sqrt(L_data)
            history['train_rmse'].append(rmse)
            history['L_data'].append(L_data)
            history['L_phys'].append(L_phys)
            history['L_mon'].append(float(L_mon))

            if X_val is not None:
                y_val_hat, _, _ = self._forward(X_val)
                val_rmse = np.sqrt(np.mean((y_val_hat - y_val)**2))
                history['val_rmse'].append(val_rmse)

            if epoch % verbose_every == 0:
                val_str = f", Val RMSE={history['val_rmse'][-1]:.2f}" if X_val is not None else ""
                print(f"  Epoch {epoch:4d}: Train RMSE={rmse:.2f} g"
                      f"  L_data={L_data:.1f}  L_phys={L_phys:.1f}"
                      f"  L_mon={float(L_mon):.2f}{val_str}")

        return history

    def predict(self, X):
        y_hat, _, _ = self._forward(X)
        return np.clip(y_hat, 0, Y_RANGE[1])

    def print_rules(self):
        """Print table of learned rules with coefficients."""
        var_names = list(VAR_LABELS.keys())
        print("\n" + "="*70)
        print("LEARNED TAKAGI-SUGENO RULES")
        print("="*70)
        for k in range(self.n_rules):
            print(f"\nRule {k+1}:")
            conds = []
            for i, vk in enumerate(var_names):
                sym, name, unit = VAR_LABELS[vk]
                lo, hi = VAR_RANGES[vk]
                c_real = self.C[k, i] * (hi - lo) + lo
                s_real = self.S[k, i] * (hi - lo)
                conds.append(f"  {sym} = {name} ∈ N({c_real:.1f}, {s_real:.1f}) [{unit}]")
            print("IF:\n" + "\n".join(conds))
            p = self.P[k]
            terms = [f"{p[0]:.3f}"]
            for i, vk in enumerate(var_names):
                sym, _, _ = VAR_LABELS[vk]
                if abs(p[i+1]) > 0.001:
                    terms.append(f"({p[i+1]:.3f})·{sym}")
            print(f"THEN: ŷ_{k+1} = " + " + ".join(terms) + " g")
        print("="*70)

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ G: ПОДГОТОВКА ДАННЫХ
# ─────────────────────────────────────────────────────────────
keys = list(VAR_RANGES.keys())
X_norm_all_2 = np.column_stack([
    (df[k].values - VAR_RANGES[k][0]) / (VAR_RANGES[k][1] - VAR_RANGES[k][0])
    for k in keys
])
Y_all_2 = df['Y_SorbentDose'].values

# Разбивка: 70% обучение, 15% валидация, 15% тест
X_tv, X_test, y_tv, y_test = train_test_split(
    X_norm_all_2, Y_all_2, test_size=0.15, random_state=RANDOM_SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.176, random_state=RANDOM_SEED)
print(f"\nSplit: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")

# ─────────────────────────────────────────────────────────────
# ЭКСПОРТ ВЫБОРОК ДЛЯ FIGSHARE
# Сохраняем все три выборки в CSV с:
#   - сырыми (raw) значениями переменных
#   - нормализованными (normalised) значениями
#   - целевой переменной Y_SorbentDose (г)
#   - меткой split (train / validation / test)
# ─────────────────────────────────────────────────────────────

def make_export_df(X_norm, y_vals, split_label):
    """Build a DataFrame with both raw and normalised columns + split label."""
    raw_cols = {}
    norm_cols = {}
    for i, (vk, (lo, hi)) in enumerate(VAR_RANGES.items()):
        raw_name  = vk                          # e.g. x1_BodySize
        norm_name = vk.replace('x', 'x_norm_', 1)  # e.g. x_norm1_BodySize
        raw_cols[raw_name]   = X_norm[:, i] * (hi - lo) + lo   # denormalise
        norm_cols[norm_name] = X_norm[:, i]
    df_out = pd.DataFrame({**raw_cols, **norm_cols})
    df_out['Y_SorbentDose_g']  = y_vals
    df_out['split']            = split_label
    return df_out

df_train_export = make_export_df(X_train, y_train, 'train')
df_val_export   = make_export_df(X_val,   y_val,   'validation')
df_test_export  = make_export_df(X_test,  y_test,  'test')

# ── Файл 1: полный датасет (все 1500 примеров) ──────────────
df_full_export = pd.concat([df_train_export, df_val_export, df_test_export],
                            ignore_index=True)
df_full_export.to_csv('PI_NFISSDR_full_dataset.csv', index=False, encoding='utf-8-sig')
print("Saved: PI_NFISSDR_full_dataset.csv  (all 1500 samples)")

# ── Файл 2: обучающая выборка (70%) ─────────────────────────
df_train_export.to_csv('PI_NFISSDR_train.csv', index=False, encoding='utf-8-sig')
print(f"Saved: PI_NFISSDR_train.csv        ({len(df_train_export)} samples, 70%)")

# ── Файл 3: валидационная выборка (15%) ─────────────────────
df_val_export.to_csv('PI_NFISSDR_validation.csv', index=False, encoding='utf-8-sig')
print(f"Saved: PI_NFISSDR_validation.csv   ({len(df_val_export)} samples, 15%)")

# ── Файл 4: тестовая выборка (15%) ──────────────────────────
df_test_export.to_csv('PI_NFISSDR_test.csv', index=False, encoding='utf-8-sig')
print(f"Saved: PI_NFISSDR_test.csv         ({len(df_test_export)} samples, 15%)")

# ── Файл 5: предсказания после обучения (заполнится позже) ──
# (сохраняется в конце файла, после predict)

print(f"""
Dataset column description:
  x1_BodySize          — bird body mass, g         [700–7000]
  x2_OilCoverage       — oil-covered feather area, % [0–100]
  x3_FeatherDensity    — effective feather density, % [0–100]
  x4_OilViscosity      — oil kinematic viscosity, mm²/s [1–40]
  x5_OilAge            — duration of oil contact, arb.u. [0–100]
  x_norm1_* … x_norm5_*— same variables normalised to [0, 1]
  Y_SorbentDose_g      — target sorbent dose, g    [0–800]
  split                — train / validation / test
""")

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ H: ОБУЧЕНИЕ PI-ANFIS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"TRAINING PI-ANFIS (rules: {N_RULES}, epochs: {N_EPOCHS})")
print(f"  λ_physics={LAMBDA_PHY}, λ_monotone={LAMBDA_MON}, lr={LR}")
print("="*60)

model = PIANFIS(n_rules=N_RULES, lr=LR,
                lambda_phy=LAMBDA_PHY, lambda_mon=LAMBDA_MON,
                n_epochs=N_EPOCHS, sigma_init=SIGMA_INIT)

history = model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose_every=50)

# Вывод обученных правил
model.print_rules()

# Предсказание
y_pred_pi = model.predict(X_test)

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ I: МОДЕЛИ СРАВНЕНИЯ
# ─────────────────────────────────────────────────────────────
print("\nTraining comparison models...")

# Random Forest
rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# SVR
svr = SVR(kernel='rbf', C=200, gamma=0.15, epsilon=3)
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)

# ANFIS without physics (same hyperparameters, λ=0)
print("Training standard ANFIS (no physics constraints)...")
model_std = PIANFIS(n_rules=N_RULES, lr=LR,
                    lambda_phy=0.0, lambda_mon=0.0,
                    n_epochs=N_EPOCHS, sigma_init=SIGMA_INIT)
history_std = model_std.fit(X_train, y_train, verbose_every=N_EPOCHS+1)  # тихий режим
y_pred_std = model_std.predict(X_test)

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ J: МЕТРИКИ
# ─────────────────────────────────────────────────────────────
def metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {'Model': name, 'RMSE (g)': round(rmse,2), 'MAE (g)': round(mae,2), 'R²': round(r2,4)}

results = [
    metrics(y_test, y_pred_pi,  'PI-ANFIS (proposed)'),
    metrics(y_test, y_pred_std, 'ANFIS (standard)'),
    metrics(y_test, y_pred_rf,  'Random Forest'),
    metrics(y_test, y_pred_svr, 'SVR (RBF)'),
]
metrics_df = pd.DataFrame(results)

print("\n" + "="*65)
print("METRICS TABLE")
print("="*65)
print(metrics_df.to_string(index=False))
print("="*65)
metrics_df.to_csv('table_metrics_v2.csv', index=False, encoding='utf-8-sig')

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ K: РИСУНКИ
# ─────────────────────────────────────────────────────────────
fig_dir = ''   # текущая папка; можно указать 'figures/'

# --- Рис.1: Функции принадлежности входных переменных ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
x_norm = np.linspace(0, 1, 300)

for idx, (vk, (lo, hi)) in enumerate(VAR_RANGES.items()):
    ax = axes[idx]
    sym, name, unit = VAR_LABELS[vk]
    x_real = x_norm * (hi - lo) + lo
    for k in range(N_RULES):
        c_real = model.C[k, idx] * (hi - lo) + lo
        s_real = model.S[k, idx] * (hi - lo)
        mu = np.exp(-0.5*((x_real - c_real)/s_real)**2)
        ax.plot(x_real, mu, color=COLORS[k], linewidth=2.2,
                label=f'Rule {k+1} (c={c_real:.0f})')
        ax.fill_between(x_real, mu, alpha=0.10, color=COLORS[k])
    ax.set_title(f'{sym} — {name}', fontsize=11, fontweight='bold', color='#1a3a6b')
    ax.set_xlabel(f'Value ({unit})')
    ax.set_ylabel('μ(x)')
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_facecolor('#f9fafb')

axes[-1].set_visible(False)
fig.suptitle('Learned Membership Functions of PI-ANFIS\n(5 rules, Gaussian MFs)',
             fontsize=13, fontweight='bold', color='#1a3a6b')
plt.tight_layout()
plt.savefig(f'{fig_dir}fig1_membership_functions_v2.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.show()
print("fig1_membership_functions_v2.png saved")

# --- Рис.2: Физические базовые функции ---
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
phy_data = [
    (np.linspace(700, 7000, 200), 'x₁ BodySize',        'g',       lambda x: (x-700)/(7000-700),          '#1a3a6b'),
    (np.linspace(0, 100, 200),   'x₂ OilCoverage',      '%',       lambda x: (x/100)**ALPHA,              '#1a7a6b'),
    (np.linspace(0, 100, 200),   'x₃ FeatherDensity',   '%',       lambda x: (x/100)**BETA,               '#c0392b'),
    (np.linspace(1, 40, 200),    'x₄ OilViscosity',     'mm²/s',   lambda x: np.log1p(x-1)/np.log1p(39), '#f39c12'),
    (np.linspace(0, 100, 200),   'x₅ OilAge',           'arb. u.', lambda x: 1-np.exp(-GAMMA*x/100),     '#7d3c98'),
]
formulas = [
    f'f₁(x)=(x−700)/6300', f'f₂(x)=(x/100)^{ALPHA}',
    f'f₃(x)=(x/100)^{BETA}', f'f₄(x)=log(1+x)/log(40)',
    f'f₅(x)=1−e^(−{GAMMA}x/100)',
]
for i, (xv, title, unit, fn, color) in enumerate(phy_data):
    y = fn(xv)
    axes[i].plot(xv, y, color=color, linewidth=2.5)
    axes[i].fill_between(xv, y, alpha=0.15, color=color)
    axes[i].set_title(title, fontsize=10, fontweight='bold', color=color)
    axes[i].set_xlabel(f'({unit})', fontsize=9)
    axes[i].set_ylabel('f(x)', fontsize=9)
    axes[i].text(0.05, 0.88, formulas[i], transform=axes[i].transAxes,
                 fontsize=8, color=color, fontstyle='italic')
    axes[i].grid(True, alpha=0.3, linestyle=':')
    axes[i].set_facecolor('#f9fafb')
    axes[i].set_ylim(0, 1.1)

fig.suptitle('Physics-Informed Monotone Base Functions fᵢ(xᵢ)',
             fontsize=13, fontweight='bold', color='#1a3a6b')
plt.tight_layout()
plt.savefig(f'{fig_dir}fig2_physics_functions_v2.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.show()
print("fig2_physics_functions_v2.png saved")

# --- Рис.3: Кривые обучения ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(len(history['train_rmse']))

axes[0].plot(epochs_range, history['train_rmse'], color='#1a3a6b', lw=2, label='Train RMSE (PI-ANFIS)')
if history['val_rmse']:
    axes[0].plot(epochs_range, history['val_rmse'], color='#c0392b', lw=2, linestyle='--', label='Val RMSE (PI-ANFIS)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('RMSE (g)')
axes[0].set_title('Training Curves (RMSE)', fontweight='bold', color='#1a3a6b')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].set_facecolor('#f9fafb')

axes[1].plot(epochs_range, history['L_data'],  color='#1a3a6b', lw=2, label='L_data')
axes[1].plot(epochs_range, history['L_phys'],  color='#1a7a6b', lw=2, label='L_physics', linestyle='--')
axes[1].plot(epochs_range, history['L_mon'],   color='#f39c12', lw=2, label='L_monotone', linestyle=':')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss value')
axes[1].set_title('Loss Function Components', fontweight='bold', color='#1a3a6b')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_facecolor('#f9fafb')

fig.suptitle('PI-NFISSDR Training Process', fontsize=13, fontweight='bold', color='#1a3a6b')
plt.tight_layout()
plt.savefig(f'{fig_dir}fig3_training_curves_v2.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.show()
print("fig3_training_curves_v2.png saved")

# --- Рис.4: Scatter plots 4 моделей ---
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
axes = axes.flatten()
all_preds = [
    ('PI-ANFIS (proposed)',   y_pred_pi,  '#1a3a6b'),
    ('ANFIS (standard)',      y_pred_std, '#1a7a6b'),
    ('Random Forest',         y_pred_rf,  '#c0392b'),
    ('SVR (RBF)',             y_pred_svr, '#f39c12'),
]
for i, (name, yp, color) in enumerate(all_preds):
    ax = axes[i]
    rmse = np.sqrt(mean_squared_error(y_test, yp))
    r2   = r2_score(y_test, yp)
    ax.scatter(y_test, yp, alpha=0.45, color=color, s=15)
    ax.plot([0,800],[0,800], 'k--', lw=1.5, alpha=0.6)
    ax.set_xlim(0,800); ax.set_ylim(0,800)
    ax.set_xlabel('Actual dose (g)'); ax.set_ylabel('Predicted dose (g)')
    ax.set_title(f'{name}\nRMSE={rmse:.1f} g,  R²={r2:.3f}',
                 fontweight='bold', color=color, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':'); ax.set_facecolor('#f9fafb')
    ax.set_aspect('equal')

fig.suptitle('Comparative Analysis: Predicted vs Actual Sorbent Dose',
             fontsize=13, fontweight='bold', color='#1a3a6b')
plt.tight_layout()
plt.savefig(f'{fig_dir}fig4_scatter_v2.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.show()
print("fig4_scatter_v2.png saved")

# --- Рис.5: Гистограммы метрик ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
model_short = ['PI-ANFIS', 'ANFIS', 'RF', 'SVR']
bar_colors = ['#1a3a6b', '#1a7a6b', '#c0392b', '#f39c12']

# Rename columns for English display
metrics_df_en = metrics_df.rename(columns={'RMSE (г)': 'RMSE (g)', 'MAE (г)': 'MAE (g)'})

for idx, metric in enumerate(['RMSE (g)', 'MAE (g)', 'R²']):
    vals = metrics_df_en[metric].values
    bars = axes[idx].bar(model_short, vals, color=bar_colors, edgecolor='white', lw=1.5, width=0.55)
    for bar, val in zip(bars, vals):
        axes[idx].text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + max(abs(v) for v in vals)*0.02,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[idx].set_title(metric, fontweight='bold', fontsize=12, color='#1a3a6b')
    axes[idx].set_ylabel(metric)
    axes[idx].grid(True, alpha=0.3, axis='y', linestyle=':')
    axes[idx].set_facecolor('#f9fafb')
    if metric == 'R²':
        axes[idx].axhline(1.0, color='green', linestyle='--', alpha=0.5)
        axes[idx].set_ylim(min(0, min(vals)-0.1), 1.15)

fig.suptitle('Model Comparison: RMSE, MAE, and R² on the Test Set',
             fontsize=13, fontweight='bold', color='#1a3a6b')
plt.tight_layout()
plt.savefig(f'{fig_dir}fig5_metrics_v2.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.show()
print("fig5_metrics_v2.png saved")

# --- Рис.6: Поверхности отклика ---
fig = plt.figure(figsize=(16, 6))
pairs = [
    (0, 1, 'x₁ BodySize (g)',      'x₂ OilCoverage (%)'),
    (2, 3, 'x₃ FeatherDensity (%)', 'x₄ OilViscosity (mm²/s)'),
]
medians = np.median(X_train, axis=0)

for p, (i1, i2, lbl1, lbl2) in enumerate(pairs):
    ax = fig.add_subplot(1, 2, p+1, projection='3d')
    g1 = np.linspace(0, 1, 25)
    g2 = np.linspace(0, 1, 25)
    G1, G2 = np.meshgrid(g1, g2)
    ZZ = np.zeros_like(G1)
    for ii in range(25):
        for jj in range(25):
            xrow = medians.copy()
            xrow[i1] = G1[ii, jj]
            xrow[i2] = G2[ii, jj]
            ZZ[ii, jj] = model.predict(xrow.reshape(1,-1))[0]

    # Денормализовать оси для наглядности
    lo1,hi1 = list(VAR_RANGES.values())[i1]
    lo2,hi2 = list(VAR_RANGES.values())[i2]
    surf = ax.plot_surface(G1*(hi1-lo1)+lo1, G2*(hi2-lo2)+lo2, ZZ,
                           cmap='viridis', alpha=0.85, edgecolor='none')
    ax.set_xlabel(lbl1, fontsize=8, labelpad=5)
    ax.set_ylabel(lbl2, fontsize=8, labelpad=5)
    ax.set_zlabel('Dose (g)', fontsize=8)
    ax.set_title(f'Response surface: {lbl1}\n× {lbl2}',
                 fontsize=9, fontweight='bold', color='#1a3a6b')
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label='g')

fig.suptitle('PI-NFISSDR Response Surfaces (other variables fixed at median)',
             fontsize=12, fontweight='bold', color='#1a3a6b')
plt.tight_layout()
plt.savefig(f'{fig_dir}fig6_response_surfaces_v2.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("fig6_response_surfaces_v2.png saved")

# --- Рис.7: Анализ чувствительности ---
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for i, (vk, (lo, hi)) in enumerate(VAR_RANGES.items()):
    sym, name, unit = VAR_LABELS[vk]
    xv = np.linspace(0, 1, 60)
    y_pd = []
    for val in xv:
        xrow = medians.copy()
        xrow[i] = val
        y_pd.append(model.predict(xrow.reshape(1,-1))[0])
    x_real = xv*(hi-lo)+lo
    axes[i].plot(x_real, y_pd, color=COLORS[i], lw=2.5)
    axes[i].fill_between(x_real, y_pd, alpha=0.15, color=COLORS[i])
    axes[i].set_title(f'{sym} {name}', fontsize=10, fontweight='bold', color=COLORS[i])
    axes[i].set_xlabel(f'({unit})', fontsize=9)
    axes[i].set_ylabel('Sorbent dose (g)', fontsize=9)
    axes[i].grid(True, alpha=0.3, linestyle=':')
    axes[i].set_facecolor('#f9fafb')
    axes[i].set_ylim(0, Y_RANGE[1])

fig.suptitle('Sensitivity Analysis: Effect of Each Input Variable on Sorbent Dose\n'
             '(all other variables fixed at median)',
             fontsize=12, fontweight='bold', color='#1a3a6b')
plt.tight_layout()
plt.savefig(f'{fig_dir}fig7_sensitivity_v2.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.show()
print("fig7_sensitivity_v2.png saved")

# ─────────────────────────────────────────────────────────────
# РАЗДЕЛ L: ИТОГ + РЕКОМЕНДАЦИИ ПО НАСТРОЙКЕ
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("FINAL SUMMARY")
print("="*65)
pi_rmse = metrics_df.loc[0,'RMSE (g)']
best_idx = metrics_df['RMSE (g)'].idxmin()
best = metrics_df.loc[best_idx]
print(f"PI-ANFIS: RMSE={pi_rmse:.2f} g,  R²={metrics_df.loc[0,'R²']:.4f}")
print(f"Best model: {best['Model']} — RMSE={best['RMSE (g)']:.2f} g")
print("="*65)
print("""
TUNING GUIDE (edit the HYPERPARAMETERS section at the top of the file):
────────────────────────────────────────────────────────────────────────
Parameter     Current  Try            Effect
────────────────────────────────────────────────────────────────────────
N_EPOCHS       500     1000..2000     ↓ RMSE with stable training
LR             0.005   0.001..0.05    Convergence speed
LAMBDA_PHY     0.3     0.1..1.0       Weight of physics constraints
LAMBDA_MON     0.1     0.0..0.5       Monotonicity penalty
SIGMA_INIT     0.25    0.15..0.40     Initial MF width
N_RULES        5       5..10          Model complexity
ALPHA          0.7     0.5..1.0       OilCoverage nonlinearity
BETA           1.5     1.0..2.5       FeatherDensity nonlinearity
GAMMA          3.0     1.0..5.0       OilAge saturation rate
────────────────────────────────────────────────────────────────────────
""")

print("\nSaved figures:")
for f in ['fig1_membership_functions_v2.png', 'fig2_physics_functions_v2.png',
          'fig3_training_curves_v2.png', 'fig4_scatter_v2.png',
          'fig5_metrics_v2.png', 'fig6_response_surfaces_v2.png',
          'fig7_sensitivity_v2.png']:
    print(f"  ✓ {f}")
print("  ✓ table_metrics_v2.csv")

# ─────────────────────────────────────────────────────────────
# ЭКСПОРТ ДАННЫХ ДЛЯ FIGSHARE (Nature requirement)
# Запускается ПОСЛЕ обучения, чтобы включить предсказания
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("EXPORTING DATASETS FOR FIGSHARE")
print("="*65)

def make_export_df(X_norm, y_true, split_label,
                   y_pi=None, y_std=None, y_rf=None, y_svr=None):
    """
    Build export DataFrame with raw + normalised columns,
    true target, and (optionally) model predictions.
    """
    cols = {}
    # Raw (physical units)
    for i, (vk, (lo, hi)) in enumerate(VAR_RANGES.items()):
        cols[vk] = np.round(X_norm[:, i] * (hi - lo) + lo, 4)
    # Normalised [0,1]
    for i, vk in enumerate(VAR_RANGES.keys()):
        cols[f'{vk}_norm'] = np.round(X_norm[:, i], 6)
    # Target
    cols['Y_SorbentDose_g'] = np.round(y_true, 2)
    # Predictions (test split only)
    if y_pi  is not None: cols['Y_pred_PI_ANFIS_g']  = np.round(y_pi,  2)
    if y_std is not None: cols['Y_pred_ANFIS_std_g']  = np.round(y_std, 2)
    if y_rf  is not None: cols['Y_pred_RandomForest_g']= np.round(y_rf,  2)
    if y_svr is not None: cols['Y_pred_SVR_g']         = np.round(y_svr, 2)
    cols['split'] = split_label
    return pd.DataFrame(cols)

# ── 1. Training set (1050 samples, 70%) ──────────────────────
df_train_exp = make_export_df(X_train, y_train, 'train')
df_train_exp.to_csv('PI_NFISSDR_train.csv', index=False, encoding='utf-8-sig')
print(f"  ✓ PI_NFISSDR_train.csv          — {len(df_train_exp)} samples (70%)")

# ── 2. Validation set (225 samples, 15%) ─────────────────────
df_val_exp = make_export_df(X_val, y_val, 'validation')
df_val_exp.to_csv('PI_NFISSDR_validation.csv', index=False, encoding='utf-8-sig')
print(f"  ✓ PI_NFISSDR_validation.csv     — {len(df_val_exp)} samples (15%)")

# ── 3. Test set (225 samples, 15%) + all model predictions ───
df_test_exp = make_export_df(
    X_test, y_test, 'test',
    y_pi=y_pred_pi, y_std=y_pred_std, y_rf=y_pred_rf, y_svr=y_pred_svr
)
df_test_exp.to_csv('PI_NFISSDR_test.csv', index=False, encoding='utf-8-sig')
print(f"  ✓ PI_NFISSDR_test.csv           — {len(df_test_exp)} samples (15%) + predictions")

# ── 4. Full dataset (all 1500 samples) ───────────────────────
df_full_exp = pd.concat([df_train_exp, df_val_exp,
                          df_test_exp.drop(columns=[c for c in df_test_exp.columns
                                                    if 'Y_pred' in c])],
                         ignore_index=True)
df_full_exp.to_csv('PI_NFISSDR_full_dataset.csv', index=False, encoding='utf-8-sig')
print(f"  ✓ PI_NFISSDR_full_dataset.csv   — {len(df_full_exp)} samples (complete)")

print(f"""
Column description (all CSV files):
  x1_BodySize            Bird body mass                  [700–7000 g]
  x2_OilCoverage         Oil-covered feather surface     [0–100 %]
  x3_FeatherDensity      Effective feather density        [0–100 %]
  x4_OilViscosity        Oil kinematic viscosity          [1–40 mm²/s]
  x5_OilAge              Duration of oil contact          [0–100 arb.u.]
  *_norm                 Same variables normalised to     [0–1]
  Y_SorbentDose_g        Target: required sorbent dose   [0–800 g]
  Y_pred_PI_ANFIS_g      Prediction: PI-ANFIS proposed   (test only)
  Y_pred_ANFIS_std_g     Prediction: standard ANFIS       (test only)
  Y_pred_RandomForest_g  Prediction: Random Forest        (test only)
  Y_pred_SVR_g           Prediction: SVR (RBF kernel)     (test only)
  split                  train / validation / test

FIGSHARE upload checklist:
  □ PI_NFISSDR_full_dataset.csv   — complete dataset (1500 rows)
  □ PI_NFISSDR_train.csv          — training split   (1050 rows)
  □ PI_NFISSDR_validation.csv     — validation split  (225 rows)
  □ PI_NFISSDR_test.csv           — test split + model predictions (225 rows)
  □ table_metrics_v2.csv          — comparative metrics table
  □ pi_anfis_v2.py                — source code (reproducibility)
""")