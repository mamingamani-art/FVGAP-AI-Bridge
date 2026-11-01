import matplotlib
matplotlib.use('Agg')   # 🆕 [FIX] رفع کرش TclError - بک‌اند غیرتعاملی، بدون نیاز به Tk/Tcl
import seaborn as sns  
import matplotlib.pyplot as plt
import joblib , logging , time ,optuna, sys, shutil
from datetime import datetime
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'      # غیرفعال کردن لاگ‌های سطح پایین و اطلاعاتی تنسورفلور
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'     # غیرفعال کردن لاگ‌های سطح پایین و اطلاعاتی تنسورفلور
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout, SpatialDropout1D, Dense, Bidirectional, BatchNormalization, Layer, Input,GaussianNoise,GlobalAveragePooling1D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
from colorama import Fore,Back,Style, init
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from flask import Flask, request, jsonify
import threading
import visualkeras
from PIL import ImageFont
from sklearn.isotonic import IsotonicRegression 
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.initializers import GlorotUniform, Zeros
from colorama import Fore, init
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_curve, auc, precision_recall_curve   # 🆕 [VIZ #4] برای ROC / Precision-Recall Curve
from keras_visualizer import visualizer

init(autoreset=True)  # مقداردهی اولیه برای رنگی کردن متن‌ها





input_n_trials = 300      #   tedade run baraye Optuna

MAX_BACKUPS_TO_KEEP = 10   # 🆕 فقط N تا از آخرین بکاپ‌های "BEST_Trial..." روی دسکتاپ نگه داشته می‌شود (بکاپ FINAL_HOLDOUT همیشه نگه داشته می‌شود و جزو این محدودیت نیست)

# 🆕 [SHADOW EVAL] اگر True باشد: در پایان ۳۰۰ Trial، اگر Shadow Holdout Log
# نشان دهد یک Trial دیگر (غیر از انتخاب رسمی Optuna که فقط Validation را
# دیده) واقعاً روی Holdout بهتر عمل کرده، مدل نهایی که در MODELS_DIR ذخیره
# می‌شود و Phase 5 (Live) از آن می‌خواند، بر اساس همان Trial واقعاً بهتر
# بازآموزی می‌شود - نه انتخاب رسمی Optuna. در هر دو حالت (True/False)، هر دو
# نسخه (انتخاب رسمی + انتخاب واقعاً بهتر، اگر متفاوت باشند) در پوشه‌های
# جداگانه و برچسب‌دار روی دسکتاپ بکاپ گرفته می‌شوند - هیچ‌چیز گم نمی‌شود.
PREFER_HOLDOUT_OVER_VAL_FOR_FINAL_MODEL = True

# 🆕 [SHADOW EVAL - FIX] سقف قابل‌قبول Drawdown روی Holdout برای این‌که یک
# Trial اصلاً وارد رقابت «بهترین واقعی» بشود - مستقل از تعداد معاملات، یک
# مدل با Drawdown فاجعه‌بار (مثلاً ۷۰-۸۵٪) هرگز نباید صرفاً به‌خاطر KPI بالا
# انتخاب شود، چون با drawdown این‌قدر بزرگ نتیجه‌ی نهایی به‌شدت به توالی
# تصادفی معاملات وابسته است، نه به مهارت واقعی مدل. عدد ۰.۳۵ (۳۵٪) یک
# آستانه‌ی محافظه‌کارانه‌ی معقول برای معاملات ۵ دقیقه‌ای با ریسک ۱٪ هر
# معامله است - در صورت نیاز می‌توانی این عدد را با توجه به تحمل ریسک خودت
# تغییر بدهی.
MAX_ACCEPTABLE_HOLDOUT_MDD = 0.35




# --------------  Base Model  --------------
epoch_BASE    = 30                    # 70   
base_patience = 5                # agar dar X epoch Loss kamtar nashod STOP mishavad
base_learning_rate = 0.00005
base_l2_reg = 0.001
class_weights_win_Mother = 1.5   # برای مدل مادر تعادل وزنی خودکار کافیست
Base_n_splits = 10               # تعداد اتاق‌ها 

base_GaussianNoise = 0.03


N1_Conv1D = 128  
N2_Bidirectional = 128  
N3_Conv1D = 256   
N4_Dense = 256  

N1_Dropout = 0.4  
N2_Dropout = 0.4
N3_Dropout = 0.5

# --------------  CAT Model  ---------------   
epoch_cat = 7                   # 20    
cat_patience = 3                 # agar dar X epoch Loss taqiir nakard  nerkhe amoozesh nesf mishavad
cat_learning_rate = 0.00005       # 0.0001

DAUGHTER_WEIGHTS_CONFIG = {     # تنظیمات وزن‌دهی اختصاصی برای هر دسته                DAUGHTER_GAIN_WEIGHT 
    '0': 1.5 ,  # وزن برای معاملات فروش     Cat 0 = SELL
    '1': 1.5    # وزن برای معاملات خرید     Cat 1 = BUY
}

daughter_n_splits = 4              # تعداد فولدهای آموزش مدل دختر (مثلاً ۳ یا ۵)
X_layeAkhar_UnFreez = 2            # tedade laye haye akhar ke unfreez beshan baraye model haye dokhtar

# تنظیمات جدید برای ساختار خانواده فیچرها (۵ خانواده × ۳۰ گام = ۱۵۰ فیچر)
TIME_STEPS = 30   # 30             # هر خانواده ۳۰ گام زمانی دارد , اگر این متغببر را کمتر از 30 بزاری همان مقدار گام ها از ان خانواده برداشته میشود
TOP_K = 20         # 13             # تعداد خانواده‌های برتر که انتخاب می‌کنیم
Darsade_behtarin_featurha = 0.5    # chand darsad az behtarin va badtarin featur hara neshen dahad

# 🆕 [FIX #3] تنظیمات تست معناداری آماری (Permutation Test) برای انتخاب فیچر داینامیک.
# به‌جای یک آستانه‌ی ثابت MI (که با تغییر رژیم بازار یا اضافه‌شدن فیچر جدید یا
# می‌تواند خیلی سخت‌گیر شود یا خیلی بی‌اثر)، هر خانواده با توزیع نویزِ خودِ همان
# دیتاست (MI روی برچسب shuffle‌شده) مقایسه می‌شود -> آستانه خودش را از داده‌ی
# فعلی می‌گیرد و با هر تغییر رژیم/دیتاست/فیچر خودکار جابه‌جا می‌شود.
MI_PERMUTATIONS = 10             # تعداد دفعات shuffle برچسب برای تخمین توزیع نویز هر خانواده
MI_SIGNIFICANCE_PERCENTILE = 95  # اگر MI واقعی از این صدک توزیع نویز بیشتر نباشد، خانواده رد می‌شود


MIN_TRADE_PERCENT = 0.03  # حداقل ۳ درصد کل موقعیت‌ها باید ترید شوند (یکسان‌سازی‌شده: هم در Phase 3 هم در calculate_unified_kpi همین یک عدد استفاده میشه)


# ================================================================
# --- تنظیمات جدید: مدل روند (Trend Model) و متا-مدل (Meta Model) ---
# ================================================================
TREND_FAMILY_PREFIX = "F[Trend"      # پیشوند مشترک خانواده‌های فیچر روند (TrendADX, TrendRSI, TrendMACD, TrendMASlope)
TREND_TARGET_COL     = "Trend_Label_Future_1"
TREND_n_splits       = 5              # تعداد فولد walk-forward برای مدل روند (کوچک‌تر از مادر چون فیچر کمتری دارد)
epoch_trend          = 15
trend_patience       = 4
trend_learning_rate  = 0.0003
trend_GaussianNoise  = 0.03

META_MODEL_MIN_ROWS  = 30             # حداقل تعداد نمونه‌ی معتبر (غیر NaN) برای آموزش متا-مدل هر دسته


# --- تنظیمات و مسیرهای دقیق پروژه ---
# ---------------- انتخاب حالت بک یا فوروارد تست ----------------
print("\n" + "File ha ( DATASET & TEST ) az kodam masir bardashte shavand ? " + "\n" + " MT4 >> tester >> Files   [ 0 ]" + "\n" + " MT4 >> MQL4   >> Files   [ 1 ]")
TEST_MODE = int (input("                            ?   "))  
# مسیرهای ورودی (متاتریدر)
if TEST_MODE != 1:
    DATASET_PATH            = r"C:\Program Files (x86)\Alpari MT4\MQL4\Experts\version 21 - FVGAP ( MIX FEATURS ) ( FIX Z ) 5minTrade ( ravand H4 )\DATASET  ( 2024.1 - 2026.1 ).csv"
    TEST_DATA_PATH          = r"C:\Program Files (x86)\Alpari MT4\MQL4\Experts\version 21 - FVGAP ( MIX FEATURS ) ( FIX Z ) 5minTrade ( ravand H4 )\TEST  ( 2026.1 - 2026.8 ).csv" 
    TRADE_PATH              = r"C:\Program Files (x86)\Alpari MT4\tester\files\TradingBridge\trade_result.csv"

    print("---------->> Back Test <<----------")
if TEST_MODE == 1:
    DATASET_PATH            = r"C:\Program Files (x86)\Alpari MT4\MQL4\Experts\version 21 - FVGAP ( MIX FEATURS ) ( FIX Z ) 5minTrade ( ravand H4 )\DATASET  ( 2024.1 - 2026.1 ).csv"
    TEST_DATA_PATH          = r"C:\Program Files (x86)\Alpari MT4\MQL4\Experts\version 21 - FVGAP ( MIX FEATURS ) ( FIX Z ) 5minTrade ( ravand H4 )\TEST  ( 2026.1 - 2026.8 ).csv"    
    TRADE_PATH              = r"C:\Program Files (x86)\Alpari MT4\MQL4\files\TradingBridge\trade_result.csv"
    print("---------->> Forward Test <<----------")


DATA_SET_MAH_AKHAR_PATH = DATASET_PATH

MODELS_DIR = r"C:\Users\User\Desktop\Trading_Models_5minTrade"

def perform_deep_data_audit(path):
    print(Fore.CYAN + f"⏳ در حال کالبدشکافی هوشمند خانواده‌های ویژگی در: {path}")
    try:
        df = pd.read_csv(path, sep=';')
        df.columns = [str(c).strip() for c in df.columns]
        df['target'] = (df['Y'] > 0).astype(int)
        
        feature_cols = df.columns[4:844]
        
        steps = 30
        family_stats = []

        for i in range(0, len(feature_cols), steps):
            family = feature_cols[i : i + steps]
            if len(family) == 0: continue
            base_name = str(family[0]).rsplit('_t', 1)[0]
            avg_corr = df[family].corrwith(df['target']).abs().mean()
            family_stats.append({'Family': base_name, 'Avg_Correlation': avg_corr})

        f_df = pd.DataFrame(family_stats).sort_values(by='Avg_Correlation', ascending=False)
        
        print(Fore.YELLOW + "\n--- 5 خانواده برتر (بر اساس میانگین همبستگی ۳۰ گام) ---")
        print(f_df.head(5))

        # --- بخش جدید: رسم نمودار برای هر ۵ خانواده برتر ---
        print(Fore.CYAN + "\n📊 در حال رسم نقشه همبستگی برای ۵ تیم برتر...")
        for i in range(min(5, len(f_df))):
            top_family_name = f_df.iloc[i]['Family']
            # پیدا کردن ستون‌هایی که با نام این خانواده شروع می‌شوند
            top_family_cols = [c for c in feature_cols if c.startswith(top_family_name)][:steps]
            
            plt.figure(figsize=(12, 2))
            family_corr = df[top_family_cols].corrwith(df['target']).abs().to_frame().T
            sns.heatmap(family_corr, annot=False, cmap='YlGnBu')
            plt.title(f"Rank {i+1}: {top_family_name} (Correlation Profile)")
            plt.show()
        # ------------------------------------------------

        max_avg_corr = f_df['Avg_Correlation'].max()
        print(Fore.MAGENTA + "\n=== نتیجه‌گیری نهایی آنالیز خانواده‌ها ===")
        if max_avg_corr < 0.015:
            print(Fore.RED + f"❌ فاجعه: میانگین همبستگی برترین خانواده {max_avg_corr:.4f} است.")
        elif max_avg_corr < 0.03:
            print(Fore.YELLOW + f"⚠️ هشدار: همبستگی متوسط ({max_avg_corr:.4f}).")
        else:
            print(Fore.GREEN + f"✅ عالی: همبستگی قدرتمند ({max_avg_corr:.4f}).")

    except Exception as e:
        print(Fore.RED + f"❌ خطا در اجرای کالبدشکافی: {e}")






# ==========================================
# --- منوی کنترل جریان اجرای هوشمند ---
# ==========================================
print(Fore.YELLOW + "\n=== SYSTEM CONFIGURATION ===")

# 1. Mother Model
print("1. Mother Model:")
train_mother = int(input("   Train (1)  |  Load Existing (0) : "))

# 2. Daughter Models
print("\n2. Daughter Models (Specialized):")
print("   0: Train Only SELL (Cat 0)")
print("   1: Train Only BUY (Cat 1)")
print("   2: Train BOTH")
print("   3: Load Existing (NONE Train)")
train_daughter = int(input("   Select (0, 1, 2, 3) : "))

# 3. Thresholds
print("\n3. Optimize Thresholds?")
print("   0: Optimize Only SELL")
print("   1: Optimize Only BUY")
print("   2: Optimize BOTH")
print("   3: Load Existing (NONE Optimize)")
th_choice = int(input("   Select (0, 1, 2, 3) : "))

# 4. Backtest
print("\n4. Phase 4: Statistical Backtest")
run_test = int(input("   Run Backtest? (1 = Yes, 0 = No) : "))

# 5. Live Mode
print("\n5. Phase 5: LIVE MODE")
LIVE_MODE = int(input("   Run LIVE Bridge? (1 = Yes, 0 = No) : "))





os.makedirs(MODELS_DIR, exist_ok=True)

BASE_MODEL_PATH = os.path.join(MODELS_DIR, "Base_Model")
CAT_MODELS_PATH = os.path.join(MODELS_DIR, "Category_Models")
SCALERS_PATH = os.path.join(MODELS_DIR, "Scalers")
TREND_MODEL_PATH = os.path.join(MODELS_DIR, "Trend_Model")
META_MODELS_PATH = os.path.join(MODELS_DIR, "Meta_Models")

for folder in [BASE_MODEL_PATH, CAT_MODELS_PATH, SCALERS_PATH, TREND_MODEL_PATH, META_MODELS_PATH]:
    os.makedirs(folder, exist_ok=True)


def build_transfer_model(base_model):
    
    
    for layer in base_model.layers:
        layer.trainable = isinstance(layer, BatchNormalization) or layer in base_model.layers[-X_layeAkhar_UnFreez:]
            
    unfreeze_layers = base_model.layers[-X_layeAkhar_UnFreez:]
    initializer = GlorotUniform() # Xavier initialization

    for layer in unfreeze_layers:
        if hasattr(layer, 'kernel'):
            # به جای صفر کردن، با مقادیر تصادفی استاندارد پر می‌کنیم
            new_weights = [initializer(layer.kernel.shape), Zeros()(layer.bias.shape) if layer.use_bias else None]
            layer.set_weights([w for w in new_weights if w is not None])
            print(f"✅ Layer {layer.name} Reset with GlorotUniform.")

    return Model(inputs=base_model.input, outputs=base_model.output)






def calibrate_output(probs, y_true):
    # پلات اسکیلینگ (Sigmoid)
    lr = LogisticRegression(penalty=None, solver='lbfgs') 
    
    p = np.asarray(probs).reshape(-1, 1)
    # اصلاح حیاتی: تبدیل مقادیر پیوسته سود/ضرر به برچسب‌های 0 و 1
    y_binary = (np.asarray(y_true) > 0).astype(int).flatten() 
    
    mask = ~np.isnan(p).flatten()
    
    # اطمینان از وجود هر دو کلاس (خرید و فروش/سود و ضرر) در داده‌های کالیبراسیون
    if len(np.unique(y_binary[mask])) < 2:
        return None
        
    lr.fit(p[mask], y_binary[mask])
    return lr



class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=8, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        return tf.matmul(weights, value)

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.separate_heads(self.query_dense(inputs), batch_size)
        key = self.separate_heads(self.key_dense(inputs), batch_size)
        value = self.separate_heads(self.value_dense(inputs), batch_size)
        att_out = self.attention(query, key, value)
        att_out = tf.transpose(att_out, perm=[0, 2, 1, 3])
        output = tf.reshape(att_out, (batch_size, -1, self.embed_dim))
        return self.combine_heads(output) # خروجی ۳ بعدی باقی می‌ماند برای لایه‌های بعد

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads})
        return config
    


def reshape_data_to_series(X_df, time_steps=None):
    try:
        ts = time_steps if time_steps is not None else TIME_STEPS
        X_array = X_df.values.astype('float32')
        # تشخیص خودکار تعداد خانواده‌ها (K) بر اساس کل فیچرها تقسیم بر گام‌های زمانی
        k_detected = X_array.shape[1] // ts
        X_reshaped = X_array.reshape((X_array.shape[0], k_detected, ts))
        return np.transpose(X_reshaped, (0, 2, 1))
    except Exception as e:
        print(Fore.RED + f"❌ Reshaping Error: {e}")
        return None



def load_and_preprocess(path):
    if not os.path.exists(path):
        print(Fore.RED + f"❌ File not found: {path}")
        return None, None, None
    print(Fore.CYAN + f"⏳ Loading: {path}")
    df = pd.read_csv(path, sep=';')
    df['YY'] = (df['Y'] > 0).astype(int)
    keys = df.iloc[:, 0:3]
    features = df.iloc[:, 4:-1] 
    y_target = df['YY']
    print(Fore.GREEN + f"✅ Data Loaded. Features detected: {features.shape[1]}")
    return keys, features, y_target




def get_top_k_features(path, k_families=TOP_K, steps=TIME_STEPS, calib_size=0):
    """
    🧬 [Advanced Engine] نسخه‌ی جدید مبتنی بر Mutual Information + mRMR
    (جایگزین نسخه‌ی قبلی که بر پایه‌ی همبستگی خطی Pearson بود).

    🛡️ Leak-Safe: دقیقاً مثل قبل، calib_size سطر انتهایی قبل از هرگونه
    محاسبه کنار گذاشته می‌شود.

    ⚡ کش شدن: چون این تابع داخل objective() است و هر Optuna trial یک‌بار
    صدا زده می‌شود، ولی رتبه‌بندی MI به k_families (پارامتری که Optuna تیون
    می‌کند) وابسته نیست، محاسبه‌ی سنگین MI فقط یک‌بار انجام و در دیسک کش
    می‌شود. بدون این کش، هر 300 trial این محاسبه را از نو تکرار می‌کرد.
    کش بر اساس calib_size نام‌گذاری می‌شود، و علاوه بر آن، fingerprint فایل
    دیتاست (mtime + اندازه‌ی فایل) هم داخل خودِ کش ذخیره می‌شود: اگر دیتاست
    عوض شده باشد (معاملات جدید اضافه شده، رژیم بازار تغییر کرده) ولی
    calib_size به‌طور تصادفی همان مقدار قبلی دربیاید، کش قدیمی خودکار
    نامعتبر تشخیص داده شده و از نو محاسبه می‌شود.

    🆕 [FIX #3] تست معناداری آماری (Permutation Test): هر خانواده علاوه بر
    امتیاز MI واقعی‌اش، با توزیع نویزِ خودِ همان دیتاست (MI روی برچسب
    shuffle‌شده، MI_PERMUTATIONS بار) مقایسه می‌شود. فقط خانواده‌هایی که MI
    واقعی‌شان از صدک MI_SIGNIFICANCE_PERCENTILE این توزیع نویز بیشتر باشد،
    وارد رقابت mRMR می‌شوند. چون این آستانه از خودِ داده‌ی فعلی مشتق می‌شود
    (نه یک عدد ثابت)، با تغییر رژیم بازار یا اضافه‌شدن فیچر جدید خودکار
    خودش را تطبیق می‌دهد.

    نکته: بر خلاف نسخه‌ی قبلی که همیشه دقیقاً k_families خانواده برمی‌گرداند،
    این نسخه به‌خاطر فیلتر معناداری آماری + mRMR (حذف خانواده‌های هم‌خط با
    هم‌بستگی نماینده بیش از 0.85) ممکن است کمتر از k_families خانواده
    برگرداند. این یک باگ نیست؛ رفتار عمدی برای حذف نویز/افزونگی است. بقیه‌ی
    پایپ‌لاین (که current_k_families را از تعداد واقعی ستون‌های برگشتی
    محاسبه می‌کند) با این رفتار مشکلی ندارد.
    """
    cache_path = os.path.join(MODELS_DIR, f"mi_family_ranking_cache_{calib_size}.pkl")

    # 🆕 [FIX #3] fingerprint فایل دیتاست برای تشخیص خودکار تغییر داده
    try:
        file_stat = os.stat(path)
        current_fingerprint = (int(file_stat.st_mtime), file_stat.st_size)
    except OSError:
        current_fingerprint = None

    cache_valid = False
    if os.path.exists(cache_path):
        try:
            cached_data = joblib.load(cache_path)
            # کش‌های ساخته‌شده با نسخه‌ی قبلی این تابع فقط 3 عنصر دارند (بدون
            # family_null_thresholds و بدون fingerprint) - آن‌ها را نامعتبر
            # در نظر می‌گیریم تا بدون کرش، از نو و با فرمت جدید ساخته شوند.
            if len(cached_data) == 5:
                sorted_by_mi, family_members, family_rep_data, family_null_thresholds, cached_fingerprint = cached_data
                if current_fingerprint is not None and cached_fingerprint == current_fingerprint:
                    cache_valid = True
        except Exception:
            cache_valid = False

    if cache_valid:
        print(Fore.CYAN + "⚡ رتبه‌بندی MI از کش لود شد (فایل دیتاست تغییر نکرده - بدون محاسبه‌ی دوباره).")
    else:
        if os.path.exists(cache_path):
            print(Fore.YELLOW + "🔄 کش قدیمی نامعتبر است (دیتاست تغییر کرده یا فرمت کش قدیمی است) - رتبه‌بندی MI از نو محاسبه می‌شود.")
        print(Fore.CYAN + f"🧬 [Advanced Engine] در حال آنالیز غیرخطی خانواده‌ها با Mutual Information + تست معناداری آماری (فقط یک‌بار، سپس کش می‌شود)...")

        df = pd.read_csv(path, sep=';')
        df.columns = [str(c).strip() for c in df.columns]

        if calib_size and calib_size > 0:
            print(Fore.CYAN + f"   🛡️ Leak-Safe: کنار گذاشتن {calib_size} سطر انتهایی (کالیبراسیون/آینده) قبل از انتخاب فیچر")
            df = df.iloc[:-calib_size].copy()

        df['target'] = (df['Y'] > 0).astype(int)

        # استخراج 840 فیچر (از ایندکس 4 تا 844) - فقط خانواده‌های اصلی معاملاتی
        feature_cols = df.columns[4:844]

        # ساخت وزن‌های زمانی (وزن بیشتر به گام‌های تازه‌تر)
        time_weights = np.exp(-np.linspace(0, 1.5, steps))
        time_weights /= time_weights.sum()  # نرمالسازی وزن‌ها

        family_scores = {}
        family_members = {}
        family_rep_data = {}  # برای بررسی هم‌خطی بین خانواده‌ها
        family_null_thresholds = {}  # 🆕 [FIX #3] آستانه‌ی معناداری آماری هر خانواده (از permutation test)

        y_true_arr = df['target'].values
        perm_rng = np.random.RandomState(42)  # برای تکرارپذیری کامل نتایج

        for i in range(0, len(feature_cols), steps):
            family_cols = feature_cols[i: i + steps]
            if len(family_cols) < steps:
                continue

            base_name = str(family_cols[0]).rsplit('_t', 1)[0]

            # محاسبه Mutual Information غیرخطی برای ۳۰ گام این خانواده
            X_fam = df[family_cols].fillna(0)
            mi_scores = mutual_info_classif(X_fam, y_true_arr, discrete_features=False, random_state=42)

            # میانگین وزن‌دار زمانی امتیازات MI
            weighted_mi_score = np.sum(mi_scores * time_weights)

            # =================================================================
            # 🆕 [FIX #3] Permutation Test: برچسب y را MI_PERMUTATIONS بار
            # درهم می‌ریزیم و همان MI را روی این نویز خالص حساب می‌کنیم تا
            # توزیع «چیزی که حتی یک فیچر کاملاً بی‌ربط هم به‌طور اتفاقی
            # می‌گیرد» را بسازیم. این آستانه ثابت نیست و از خودِ داده‌ی فعلی
            # مشتق می‌شود، پس با رژیم بازار/فیچرهای جدید خودکار جابه‌جا می‌شود.
            # =================================================================
            null_scores = np.empty(MI_PERMUTATIONS, dtype='float64')
            for perm_i in range(MI_PERMUTATIONS):
                y_shuffled = perm_rng.permutation(y_true_arr)
                mi_null = mutual_info_classif(X_fam, y_shuffled, discrete_features=False, random_state=42)
                null_scores[perm_i] = np.sum(mi_null * time_weights)
            null_threshold = float(np.percentile(null_scores, MI_SIGNIFICANCE_PERCENTILE))

            family_scores[base_name] = weighted_mi_score
            family_members[base_name] = family_cols.tolist()
            family_rep_data[base_name] = df[family_cols[0]]  # نماینده خانواده برای سنجش تشابه
            family_null_thresholds[base_name] = null_threshold

        sorted_by_mi = sorted(family_scores.items(), key=lambda x: x[1], reverse=True)

        joblib.dump((sorted_by_mi, family_members, family_rep_data, family_null_thresholds, current_fingerprint), cache_path)
        print(Fore.GREEN + f"✅ رتبه‌بندی MI + آستانه‌های معناداری آماری محاسبه و کش شد: {cache_path}")

    # --- ساخت گزارش (سازگار با نسخه‌ی قبلی، برای مقایسه‌ی چشمی) ---
    total_families = len(sorted_by_mi)
    top_20_count = max(1, int(total_families * Darsade_behtarin_featurha))

    def _sig_tag(name, score):
        # 🆕 [FIX #3] برچسب معناداری آماری برای نمایش در گزارش
        return "Significant" if score > family_null_thresholds.get(name, float('inf')) else "Noise-level (Not Significant)"

    report_lines = []
    report_lines.append("=" * 55)
    report_lines.append("💎 TOP  " + str(Darsade_behtarin_featurha) + " %  BEST FEATURE FAMILIES (Mutual Information) 💎")
    report_lines.append("=" * 55)
    for name, score in sorted_by_mi[:top_20_count]:
        report_lines.append(f"✅ {name:<35} | Weighted MI Score: {score:.5f} | {_sig_tag(name, score)}")

    report_lines.append("\n" + "=" * 55)
    report_lines.append("🗑️ BOTTOM  " + str(Darsade_behtarin_featurha) + " %   WORST FEATURE FAMILIES (Mutual Information) 🗑️")
    report_lines.append("=" * 55)
    for name, score in sorted_by_mi[-top_20_count:]:
        report_lines.append(f"❌ {name:<35} | Weighted MI Score: {score:.5f} | {_sig_tag(name, score)}")

    report_text = "\n".join(report_lines)
    print(Fore.YELLOW + report_text)

    report_path = os.path.join(MODELS_DIR, "Feature_Families_Report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(Fore.GREEN + f"✅ گزارش کامل خانواده‌ها در این مسیر ذخیره شد:\n   {report_path}")

    # =========================================================================
    # 🆕 [VIZ #1] نمودار میله‌ای امتیاز MI همه‌ی خانواده‌ها + رنگ‌بندی معناداری.
    # این نمودار همون اطلاعات Feature_Families_Report.txt رو به‌صورت بصری نشون
    # می‌ده: هر میله = یک خانواده، طول میله = امتیاز MI وزن‌دار، رنگ سبز یعنی
    # از Permutation Test عبور کرده (Significant)، رنگ خاکستری یعنی سطح نویز.
    # علامت | مشکی روی هر میله هم آستانه‌ی نویز همون خانواده رو نشون می‌ده تا
    # فاصله‌ی "امتیاز واقعی" تا "سطح نویز" هم قابل مقایسه باشد.
    # =========================================================================
    try:
        fam_names_plot = [name for name, _ in sorted_by_mi]
        fam_scores_plot = [score for _, score in sorted_by_mi]
        fam_colors_plot = ['#2ed573' if score > family_null_thresholds.get(name, float('inf')) else '#a4a4a4'
                            for name, score in sorted_by_mi]
        fam_thresh_plot = [family_null_thresholds.get(name, np.nan) for name, _ in sorted_by_mi]

        fig_h = max(4, 0.28 * len(fam_names_plot))
        plt.figure(figsize=(10, fig_h))
        y_pos = np.arange(len(fam_names_plot))
        plt.barh(y_pos, fam_scores_plot, color=fam_colors_plot, label='Weighted MI Score')
        plt.scatter(fam_thresh_plot, y_pos, marker='|', s=200, color='black', zorder=3, label='Noise Threshold (Permutation)')
        plt.yticks(y_pos, fam_names_plot, fontsize=7)
        plt.gca().invert_yaxis()  # بالاترین امتیاز بالای نمودار قرار می‌گیرد
        plt.xlabel('Weighted Mutual Information Score')
        plt.title('Feature Family Ranking (Green=Significant, Gray=Noise-level)')
        plt.legend(loc='lower right', fontsize=8)
        plt.tight_layout()
        mi_chart_path = os.path.join(MODELS_DIR, "Feature_Families_MI_Chart.jpg")
        plt.savefig(mi_chart_path, dpi=150)
        plt.close()
        print(Fore.GREEN + f"✅ نمودار امتیاز MI خانواده‌ها ذخیره شد: {mi_chart_path}")
    except Exception as e:
        print(Fore.RED + f"⚠️ خطا در رسم نمودار MI خانواده‌ها: {e}. رد شد، بدون کرش.")

    # =========================================================================
    # 🆕 [FIX #3] فیلتر معناداری آماری قبل از mRMR: فقط خانواده‌هایی که از تست
    # permutation عبور کرده‌اند (MI واقعی‌شان به‌طور معنادار از سطح نویز
    # بیشتر است) وارد رقابت انتخاب فیچر می‌شوند. چون family_null_thresholds
    # از خودِ همین دیتاست/رژیم فعلی بازار محاسبه شده، این فیلتر با تغییر
    # بازار یا فیچرهای جدید خودکار خودش را تطبیق می‌دهد - نه یک عدد هاردکد.
    # =========================================================================
    significant_names = [name for name, score in sorted_by_mi if score > family_null_thresholds.get(name, float('inf'))]

    if len(significant_names) == 0:
        print(Fore.RED + "⚠️ هیچ خانواده‌ای از تست معناداری آماری عبور نکرد (رویداد غیرمنتظره). به‌عنوان fallback ایمن، فیلتر غیرفعال شد و از رتبه‌بندی کامل استفاده می‌شود.")
        sorted_by_mi_significant = sorted_by_mi
    else:
        insignificant_names = [name for name, _ in sorted_by_mi if name not in significant_names]
        if insignificant_names:
            print(Fore.YELLOW + f"🧪 Permutation Test: {len(insignificant_names)} خانواده به‌خاطر عدم معناداری آماری (MI ≈ سطح نویز) از رقابت انتخاب فیچر حذف شدند: {', '.join(insignificant_names)}")
        else:
            print(Fore.GREEN + "🧪 Permutation Test: همه‌ی خانواده‌ها معنادار تشخیص داده شدند.")
        sorted_by_mi_significant = [(name, score) for name, score in sorted_by_mi if name in significant_names]

    # --- انتخاب mRMR: حذف خانواده‌های هم‌خط با خانواده‌های قبلاً انتخاب‌شده ---
    selected_families = []
    for fam_name, score in sorted_by_mi_significant:
        if len(selected_families) >= k_families:
            break

        if len(selected_families) == 0:
            selected_families.append(fam_name)
        else:
            is_redundant = False
            for selected_fam in selected_families:
                corr_with_selected = abs(family_rep_data[fam_name].corr(family_rep_data[selected_fam]))
                if corr_with_selected > 0.85:  # اگر بیش از ۸۵٪ شبیه خانواده قبلی بود، رد می‌شود
                    is_redundant = True
                    break

            if not is_redundant:
                selected_families.append(fam_name)

    # --- استخراج لیست نهایی ستون‌ها برای بازگرداندن ---
    selected_columns = []
    for fam_name in selected_families:
        selected_columns.extend(family_members[fam_name])

    print(Fore.CYAN + f"🎯 [mRMR + MI + Significance] انتخاب {len(selected_columns)} ویژگی (از {len(selected_families)} خانواده‌ی برتر، معنادار و غیرتکراری از میان {len(sorted_by_mi_significant)} خانواده‌ی معنادار (از کل {total_families})؛ درخواست اولیه: {k_families}) برای آموزش تکمیل شد.")

    # =========================================================================
    # 🆕 [VIZ #2] Heatmap همبستگی بین خانواده‌های نهایی انتخاب‌شده (بعد از mRMR).
    # چون mRMR خانواده‌هایی با |correlation| > 0.85 نسبت به خانواده‌های قبلاً
    # انتخاب‌شده را حذف می‌کند، این Heatmap تاییدیه‌ی بصریه که خروجی نهایی
    # واقعاً افزونگی کمی دارد (باید بیشتر رنگ‌های نزدیک صفر/روشن دیده شود، نه
    # تیره - رنگ تیره یعنی دو خانواده هنوز به هم خیلی شبیه‌اند).
    # =========================================================================
    try:
        if len(selected_families) >= 2:
            rep_df = pd.DataFrame({fam: family_rep_data[fam] for fam in selected_families})
            corr_matrix = rep_df.corr()

            plt.figure(figsize=(max(6, 0.5 * len(selected_families)), max(5, 0.5 * len(selected_families))))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
                        square=True, cbar_kws={'label': 'Correlation'}, annot_kws={"size": 7})
            plt.title('Correlation Between Final Selected Families (Post-mRMR)')
            plt.xticks(fontsize=7, rotation=45, ha='right')
            plt.yticks(fontsize=7, rotation=0)
            plt.tight_layout()
            corr_chart_path = os.path.join(MODELS_DIR, "Selected_Families_Correlation_Heatmap.jpg")
            plt.savefig(corr_chart_path, dpi=150)
            plt.close()
            print(Fore.GREEN + f"✅ Heatmap همبستگی خانواده‌های نهایی ذخیره شد: {corr_chart_path}")
        else:
            print(Fore.YELLOW + "ℹ️ کمتر از ۲ خانواده انتخاب شده؛ رسم Heatmap همبستگی معنا ندارد.")
    except Exception as e:
        print(Fore.RED + f"⚠️ خطا در رسم Heatmap همبستگی خانواده‌های نهایی: {e}. رد شد، بدون کرش.")

    return selected_columns


# ================================================================
# --- تابع جدید: استخراج خانواده‌های فیچر روند (Trend Families) ---
# ================================================================
def get_trend_features(path, steps=TIME_STEPS):
    """
    این تابع خانواده‌های جدید روند (F[TrendADX], F[TrendRSI], F[TrendMACD],
    F[TrendMASlope]) را از فایل دیتاست پیدا و برمی‌گرداند. این خانواده‌ها کاملاً
    مستقل از خانواده‌های اصلی معاملاتی (F[A]..F[BB]) هستند و فقط برای مدل روند
    استفاده می‌شوند.
    اگر این ستون‌ها در دیتاست وجود نداشته باشند (مثلاً هنوز EA آپدیت نشده)،
    یک لیست خالی برمی‌گرداند تا کد کرش نکند و مدل روند به‌سادگی skip شود.
    """
    df = pd.read_csv(path, sep=';', nrows=1)
    df.columns = [str(c).strip() for c in df.columns]

    trend_cols = [c for c in df.columns if c.startswith(TREND_FAMILY_PREFIX)]

    if len(trend_cols) == 0:
        print(Fore.YELLOW + "⚠️ ستون‌های خانواده‌ی روند (F[Trend...]) در دیتاست پیدا نشدند. مدل روند غیرفعال می‌ماند.")
        return []

    if TREND_TARGET_COL not in df.columns:
        print(Fore.YELLOW + f"⚠️ ستون هدف روند ({TREND_TARGET_COL}) در دیتاست پیدا نشد. مدل روند غیرفعال می‌ماند.")
        return []

    print(Fore.GREEN + f"✅ {len(trend_cols)} ستون خانواده‌ی روند پیدا شد ({len(trend_cols)//steps} خانواده).")
    return trend_cols



def build_meta_features(daughter_probs, trend_probs):
    """
    ساخت فیچرهای ورودی متا-مدل: علاوه بر دو خروجی خام (دختر، روند)، دو فیچر
    تعاملی هم اضافه می‌شود تا متا-مدل (که خطی/LogisticRegression است) بتواند
    «توافق» یا «اختلاف‌نظر» بین دو مدل را هم صریحاً ببیند، نه فقط جمع خطی ساده.
    """
    daughter_probs = np.asarray(daughter_probs, dtype='float32')
    trend_probs = np.asarray(trend_probs, dtype='float32')
    agreement = daughter_probs * trend_probs
    disagreement = np.abs(daughter_probs - trend_probs)
    return np.column_stack([daughter_probs, trend_probs, agreement, disagreement])




def visualize_and_save_model(model, model_name, save_path):
    """
    🆕 [FIX] غیرفعال شد - keras_visualizer با معماری Attention سفارشی این پروژه
    و نسخه‌ی فعلی Keras ناسازگار است (چند خطای متفاوت پشت‌سرهم). این ویژگی صرفاً
    تزئینی بود (گراف بصری معماری) و هیچ تاثیری روی آموزش/KPI نداشت.
    """
    return




# ================================================================
# --- معماری و آموزش مدل روند (با جمع‌آوری هم‌زمان OOF برای متا-مدل) ---
# ================================================================
def build_trend_model(n_families, time_steps):
    """
    معماری کوچک‌تر از مدل مادر (چون فقط ۴ خانواده به‌جای ۲۸-۲۰ خانواده دارد).
    هدف این مدل صرفاً تشخیص جهت روند بازار است، نه پیش‌بینی سودآوری معامله.
    """
    inputs = Input(shape=(time_steps, n_families))
    x = GaussianNoise(trend_GaussianNoise)(inputs)
    x = Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.1))(x)
    x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=trend_learning_rate),
        loss=BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    # صدا زدن تابع رسم برای این مدل
    visualize_and_save_model(model, model_name="trend_model", save_path=MODELS_DIR)

    return model


def train_trend_model_with_oof(X_trend_df, y_trend, n_splits=TREND_n_splits):
    """
    آموزش مدل روند با walk-forward (TimeSeriesSplit) و هم‌زمان جمع‌آوری
    پیش‌بینی‌های Out-of-Fold (OOF) برای استفاده به‌عنوان ورودی متا-مدل.
    هر بلوک اعتبارسنجی (val_idx) توسط مدلی پیش‌بینی می‌شود که آن بلوک را
    در آموزش ندیده است -> leak-safe برای متا-مدل.
    """
    n_families = X_trend_df.shape[1] // TIME_STEPS
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_trend_df)
    X_3d = reshape_data_to_series(pd.DataFrame(X_scaled), time_steps=TIME_STEPS)
    y_arr = np.asarray(y_trend).astype('float32')

    oof_preds = np.full(len(y_arr), np.nan, dtype='float32')

    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = build_trend_model(n_families, TIME_STEPS)

    fold_no = 1
    for train_idx, val_idx in tscv.split(X_3d):
        print(Fore.CYAN + f"   ∟ 🌊 Trend Model Fold {fold_no}/{n_splits}...")
        X_tr, y_tr = X_3d[train_idx], y_arr[train_idx]
        X_val, y_val = X_3d[val_idx], y_arr[val_idx]

        if len(np.unique(y_tr)) < 2:
            print(Fore.YELLOW + f"   ⚠️ Fold {fold_no}: فقط یک کلاس در train وجود دارد، این فولد رد می‌شود.")
            fold_no += 1
            continue

        early_stop = EarlyStopping(monitor='val_loss', patience=trend_patience, restore_best_weights=True)
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epoch_trend,
            batch_size=32,
            verbose=1,
            callbacks=[early_stop]
        )
        # --- جمع‌آوری OOF: پیش‌بینی روی همین بلوک val_idx با مدلی که این بلوک را ندیده ---
        fold_preds = model.predict(X_val, batch_size=256, verbose=0).flatten()
        oof_preds[val_idx] = fold_preds
        print(Fore.WHITE + f"   ✅ Fold {fold_no} done. Val samples: {len(val_idx)}")
        fold_no += 1


    
    os.makedirs(TREND_MODEL_PATH, exist_ok=True)                # این خط برای اطمینان از وجود پوشه

    model.save(os.path.join(TREND_MODEL_PATH, "trend_model.keras"))
    joblib.dump(scaler, os.path.join(TREND_MODEL_PATH, "trend_scaler.pkl"))
    joblib.dump(list(X_trend_df.columns), os.path.join(TREND_MODEL_PATH, "trend_features.pkl"))
    print(Fore.GREEN + "⭐ Trend Model Trained & Saved.")

    return oof_preds, model, scaler


def prepare_trend_oof_for_training(dataset_path, calib_size):
    """
    خواندن کل دیتاست، جدا کردن همان بخش train-only (مثل مادر/دختر با کنار گذاشتن
    calib_size سطر انتهایی)، فیلتر کردن ردیف‌های دارای فیچر/لیبل روند معتبر،
    آموزش مدل روند با OOF، و برگرداندن یک آرایه‌ی هم‌طول با کل بخش train-only
    (NaN برای ردیف‌هایی که داده‌ی روند معتبر ندارند) تا مستقیماً با ایندکس‌های
    مطلقی که مدل‌های دختر استفاده می‌کنند هم‌تراز باشد.
    """
    trend_cols = get_trend_features(dataset_path)
    if len(trend_cols) == 0:
        return None

    df_full_raw = pd.read_csv(dataset_path, sep=';')
    df_full_raw.columns = [str(c).strip() for c in df_full_raw.columns]
    df_full_raw = df_full_raw.iloc[:-calib_size].copy()          # همون train-only slice مثل مادر/دختر
    df_full_raw = df_full_raw.reset_index(drop=False)            # ستون 'index' = موقعیت مطلق 0..(n-calib_size-1)

    X_trend_raw = df_full_raw[trend_cols]
    y_trend_raw = df_full_raw[TREND_TARGET_COL] if TREND_TARGET_COL in df_full_raw.columns else None

    if y_trend_raw is None:
        print(Fore.YELLOW + f"⚠️ ستون هدف روند ({TREND_TARGET_COL}) پیدا نشد. مدل روند غیرفعال می‌ماند.")
        return None

    valid_mask = X_trend_raw.notna().all(axis=1) & y_trend_raw.notna()
    n_valid = int(valid_mask.sum())
    print(Fore.CYAN + f"📊 داده‌ی معتبر برای مدل روند: {n_valid} از {len(df_full_raw)} ردیف (train-only).")

    if n_valid < META_MODEL_MIN_ROWS:
        print(Fore.YELLOW + f"⚠️ داده‌ی معتبر روند خیلی کم است ({n_valid} < {META_MODEL_MIN_ROWS}). مدل روند و متا-مدل غیرفعال می‌مانند.")
        return None

    X_trend_valid = X_trend_raw[valid_mask].reset_index(drop=True)
    y_trend_valid = y_trend_raw[valid_mask].reset_index(drop=True)
    abs_positions = df_full_raw['index'][valid_mask.values].values

    oof_trend_local, trend_model, trend_scaler = train_trend_model_with_oof(X_trend_valid, y_trend_valid)

    # =========================================================================
    # 🆕 [FIX #1] کالیبراسیون مدل روند
    # چون build_trend_model از BinaryCrossentropy(label_smoothing=0.1) استفاده
    # می‌کند، خروجی سیگموید خام این مدل سیستماتیک به‌سمت ۰.۵ فشرده می‌شود و
    # «احتمال واقعی» نیست. از طرفی build_meta_features این خروجی را در
    # agreement/disagreement به‌صورت غیرخطی مصرف می‌کند، پس نمی‌شود اعوجاج را
    # فقط با ضریب خطی متا-مدل جبران کرد. اینجا یک کالیبراتور Platt-Scaling
    # روی OOF (leak-safe) می‌سازیم، همین‌جا هم روی oof_trend_local اعمالش
    # می‌کنیم (که بعداً ورودی متا-مدل می‌شود)، و هم آن را ذخیره می‌کنیم تا
    # predict_trend_probability() در Phase 3/4/5/Live دقیقاً همین کالیبراتور
    # را روی خروجی خام مدل روند اعمال کند -> ورودی متا-مدل در train و
    # inference دقیقاً روی یک مقیاس می‌ماند.
    # =========================================================================
    trend_calibrator = calibrate_output(oof_trend_local, y_trend_valid)
    if trend_calibrator is not None:
        joblib.dump(trend_calibrator, os.path.join(TREND_MODEL_PATH, "trend_calibrator.pkl"))
        valid_local_mask = ~np.isnan(oof_trend_local)
        if valid_local_mask.any():
            oof_trend_local = oof_trend_local.copy()
            oof_trend_local[valid_local_mask] = trend_calibrator.predict_proba(
                oof_trend_local[valid_local_mask].reshape(-1, 1)
            )[:, 1]
        print(Fore.GREEN + f"✅ Trend calibrator از روی OOF ساخته و ذخیره شد ({int(valid_local_mask.sum())} ردیف معتبر).")
    else:
        print(Fore.YELLOW + "⚠️ کالیبراسیون مدل روند ممکن نشد (فقط یک کلاس در OOF). خروجی خام (فشرده به‌سمت ۰.۵) استفاده می‌شود.")

    # =========================================================================
    # 🆕 [VIZ #8] چک صداقت (Reliability Diagram) مخصوص مدل روند - دقیقاً همان
    # تحلیلی که برای مدل‌های دختر انجام می‌شود، این‌جا هم روی OOF کالیبره‌شده‌ی
    # مدل روند اجرا می‌شود تا مشخص شود آیا خروجی مدل روند بعد از کالیبراسیون
    # واقعاً «صادقانه» است یا هنوز به سمت ۰.۵ فشرده مانده.
    # =========================================================================
    try:
        valid_check_mask = ~np.isnan(oof_trend_local)
        if valid_check_mask.sum() >= 20:
            run_honesty_check(y_trend_valid.values[valid_check_mask], oof_trend_local[valid_check_mask], "Trend")
        else:
            print(Fore.YELLOW + "ℹ️ داده‌ی OOF معتبر کافی برای چک صداقت مدل روند نیست؛ رد شد.")
    except Exception as e:
        print(Fore.RED + f"⚠️ خطا در چک صداقت مدل روند: {e}. رد شد، بدون کرش.")

    oof_trend_full = np.full(len(df_full_raw), np.nan, dtype='float32')
    oof_trend_full[abs_positions] = oof_trend_local
    joblib.dump(oof_trend_full, os.path.join(TREND_MODEL_PATH, "oof_trend_full.pkl"))

    return oof_trend_full


# ================================================================
# --- توابع کمکی برای Inference: پیش‌بینی روند + ترکیب با متا-مدل ---
# ================================================================
def predict_trend_probability(df_rows):
    """
    اگر مدل/اسکیلر روند و ستون‌های لازم موجود باشند، برای هر ردیف از df_rows
    احتمال روند را برمی‌گرداند؛ در غیر این صورت None (یعنی مدل روند در دسترس
    نیست و باید بدون آن ادامه داد).
    خروجی: آرایه‌ای هم‌طول با df_rows که مقادیر بدون داده‌ی معتبر NaN هستند.
    """
    trend_model_file = os.path.join(TREND_MODEL_PATH, "trend_model.keras")
    trend_scaler_file = os.path.join(TREND_MODEL_PATH, "trend_scaler.pkl")
    trend_feats_file = os.path.join(TREND_MODEL_PATH, "trend_features.pkl")

    if not (os.path.exists(trend_model_file) and os.path.exists(trend_scaler_file) and os.path.exists(trend_feats_file)):
        return None

    try:
        trend_cols = joblib.load(trend_feats_file)
        missing_cols = [c for c in trend_cols if c not in df_rows.columns]
        if missing_cols:
            print(Fore.YELLOW + f"⚠️ ستون‌های روند در این دیتاست پیدا نشدند ({len(missing_cols)} ستون گم). پیش‌بینی روند رد می‌شود.")
            return None

        X_trend_raw = df_rows[trend_cols]
        valid_mask = X_trend_raw.notna().all(axis=1).values

        out = np.full(len(df_rows), np.nan, dtype='float32')
        if valid_mask.sum() == 0:
            return out

        trend_scaler = joblib.load(trend_scaler_file)
        trend_model = tf.keras.models.load_model(trend_model_file, custom_objects={'MultiHeadSelfAttention': MultiHeadSelfAttention})

        n_families = len(trend_cols) // TIME_STEPS
        X_valid_scaled = trend_scaler.transform(X_trend_raw[valid_mask])
        X_valid_3d = reshape_data_to_series(pd.DataFrame(X_valid_scaled), time_steps=TIME_STEPS)

        preds = trend_model.predict(X_valid_3d, batch_size=256, verbose=0).flatten()

        # 🆕 [FIX #1] اعمال همان کالیبراتوری که در prepare_trend_oof_for_training
        # (روی OOF) ساخته شده، تا خروجی این تابع دقیقاً همان مقیاسی باشد که
        # متا-مدل موقع آموزش دیده است.
        trend_cal_path = os.path.join(TREND_MODEL_PATH, "trend_calibrator.pkl")
        if os.path.exists(trend_cal_path):
            try:
                trend_calibrator = joblib.load(trend_cal_path)
                preds = trend_calibrator.predict_proba(preds.reshape(-1, 1))[:, 1]
            except Exception as cal_e:
                print(Fore.YELLOW + f"⚠️ خطا در اعمال کالیبراتور روند: {cal_e}. خروجی خام (کالیبره‌نشده) استفاده می‌شود.")

        out[valid_mask] = preds
        return out
    except Exception as e:
        print(Fore.RED + f"❌ خطا در پیش‌بینی مدل روند: {e}")
        return None


def combine_daughter_and_trend(daughter_probs, trend_probs, cat_id):
    """
    اگر متا-مدل برای این دسته (cat_id) ذخیره شده باشد و پیش‌بینی روند هم موجود
    باشد، خروجی نهایی از ترکیب [daughter_prob, trend_prob] از طریق متا-مدل
    محاسبه می‌شود. در غیر این صورت (نبود متا-مدل یا نبود مدل روند)، دقیقاً
    خروجی خام مدل دختر بدون تغییر برگردانده می‌شود -> بدون کرش، سازگار با عقب.
    """
    meta_path = os.path.join(META_MODELS_PATH, f"meta_model_{cat_id}.pkl")

    daughter_probs = np.asarray(daughter_probs, dtype='float32')

    if trend_probs is None or not os.path.exists(meta_path):
        return daughter_probs, False

    try:
        meta_model = joblib.load(meta_path)
        trend_probs = np.asarray(trend_probs, dtype='float32')

        meta_X = build_meta_features(daughter_probs, trend_probs)
        valid_mask = ~np.isnan(meta_X).any(axis=1)

        final_probs = daughter_probs.copy()
        if valid_mask.sum() > 0:
            final_probs[valid_mask] = meta_model.predict_proba(meta_X[valid_mask])[:, 1]
        return final_probs, True
    except Exception as e:
        print(Fore.RED + f"❌ خطا در ترکیب متا-مدل (Cat {cat_id}): {e}. خروجی خام مدل دختر استفاده می‌شود.")
        return daughter_probs, False


def run_honesty_check(y_true, y_probs, label):
    """
    🆕 [VIZ #8] این تابع قبلاً فقط داخل objective() تعریف شده بود و فقط برای
    مدل‌های دختر (SELL/BUY) در پایان Phase 4 صدا زده می‌شد. حالا به سطح ماژول
    منتقل شده تا از prepare_trend_oof_for_training() هم بشه صداش زد و همون
    Reliability Diagram رو برای مدل روند (Trend Model) هم ساخت - چون مدل روند
    هم از BinaryCrossentropy با label_smoothing استفاده می‌کند و دقیقاً همان
    ریسک «فشرده‌شدن خروجی به سمت ۰.۵» را دارد که مدل‌های دختر داشتند.
    پارامتر `label` جایگزین `cat_id` قبلی شده تا هم برای دسته‌ها (مثلاً "0"،
    "1") و هم برای مدل روند (مثلاً "Trend") قابل استفاده باشد.
    """
    print(Fore.CYAN + f"\n🔍 در حال اجرای تست صداقت (Reliability Check) برای {label}...")

    y_binary = (np.asarray(y_true) > 0).astype(int)

    try:
        prob_true, prob_pred = calibration_curve(y_binary, y_probs, n_bins=10)
    except Exception as e:
        print(Fore.RED + f"⚠️ خطا در محاسبه‌ی Calibration Curve برای {label}: {e}. رد شد، بدون کرش.")
        return

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration (Target)')
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, color='#1e90ff', label=f'{label} Model')
    plt.title(f'Reliability Diagram ({label})\nHow honest is the AI probability?')
    plt.xlabel('Predicted Probability (What AI says)')
    plt.ylabel('Actual Win Rate (What Market does)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    chart_path = os.path.join(MODELS_DIR, f"Reliability_Check_{label}.jpg")
    plt.savefig(chart_path, dpi=150)
    # 🛡️ مثل بقیه‌ی نمودارها: در حین Optuna (۳۰۰ Trial) plt.show() فراخوانی
    # نمی‌شود تا اجرای خودکار مسدود (block) نشود؛ فقط در اجرای دستی (run_optuna=0) نمایش داده می‌شود.
    if run_optuna == 0:
        plt.show()
    else:
        plt.close('all')

    print(f"--- Analysis Report for {label} ---")
    for i in range(len(prob_pred)):
        diff = prob_true[i] - prob_pred[i]
        status = "Overconfident (Lying) ❌" if diff < -0.05 else ("Underconfident (Shy) ⚠️" if diff > 0.05 else "Honest ✅")
        print(f"Range ~{prob_pred[i]:.2f}: Actual WR: {prob_true[i]:.2f} | Status: {status}")


def calculate_unified_kpi(d):
    """
    🆕 [SHADOW EVAL] این تابع قبلاً داخل Phase 4 (تودرتوی objective()) تعریف
    شده بود. به سطح ماژول منتقل شد تا هم Phase 4 و هم evaluate_dataset_kpi
    (که برای ارزیابی Shadow روی Holdout استفاده می‌شود) دقیقاً از یک فرمول
    KPI استفاده کنند - در غیر این صورت ریسک داشت که این دو معیار به‌مرور
    زمان از هم جدا بیفتند و مقایسه‌ی Val KPI با Holdout KPI بی‌معنا بشود.
    فرمول و منطق دقیقاً همان چیزی است که قبلاً بود، هیچ تغییری نکرده.
    """
    ai_tr = d['ai_tr']
    base_tr = d['base_tr']

    # ۱. بررسی شرط جریمه (معاملات کمتر از MIN_TRADE_PERCENT کل معاملات بازار)
    if base_tr > 0 and ai_tr < (base_tr * MIN_TRADE_PERCENT):
        return -1.0 * ((base_tr * MIN_TRADE_PERCENT) - ai_tr)

    # ۲. محاسبه فرمول اصلی KPI
    pp = (d['ai_bal'] - 1000.0) / 1000.0
    mdd_val = d['ai_mdd'] if d['ai_mdd'] > 0.03 else 0.03

    # بردهای مدل نسبت به کل بردهای موجود در بازار
    ratio_win = (d['ai_wins'] / d['base_wins']) if d['base_wins'] > 0 else 0.0
    # ضررهای فرارشده نسبت به کل ضررهای موجود در بازار
    ratio_loss = (d['saved_loss'] / d['pot_loss']) if d['pot_loss'] > 0 else 0.0

    return (pp / mdd_val) * (ratio_win ** 2) * ratio_loss


def evaluate_dataset_kpi(df_eval, best_thresholds):
    """
    🆕 [SHADOW EVAL] نسخه‌ی سبک و مستقل از منطق Phase 4 - فقط برای محاسبه‌ی
    KPI روی یک دیتاست دلخواه (این‌جا برای Final-Test/Holdout استفاده می‌شود)
    بدون رسم نمودار و بدون نوشتن گزارش فایل، چون قراره در هر Trial (تا ۳۰۰
    بار) صدا زده شود و ساختن ۳۰۰ نمودار/گزارش اضافه فقط دیسک را شلوغ می‌کند
    بدون فایده‌ی عملی.
    عمداً به‌جای رفکتور کردن Phase 4، به‌صورت مستقل نوشته شده تا صفر ریسک
    روی منطقی که همین الان درست کار می‌کند وارد شود؛ threshold ها را از
    بیرون (best_thresholds که قبلاً در Phase 4 لود/ساخته شده) می‌گیرد - یعنی
    خودش threshold جدیدی روی این دیتاست تعیین نمی‌کند (چون اگر این کار را
    می‌کرد، خودِ Holdout هم می‌شد بخشی از فرایند انتخاب و دیگر «ندیده» نبود).
    خروجی: (kpi_score, d) - اگر مدل/داده‌ی کافی نبود، (None, d) برمی‌گرداند.
    """
    y_real_eval = df_eval.iloc[:, 3].values
    cat_ids_eval = df_eval.iloc[:, 0].astype(int).values
    total_rows_eval = len(df_eval)

    d = {'base_bal': 1000.0, 'ai_bal': 1000.0, 'base_wins': 0, 'ai_wins': 0,
         'base_tr': 0, 'ai_tr': 0, 'base_peak': 1000.0, 'ai_peak': 1000.0,
         'base_mdd': 0, 'ai_mdd': 0, 'saved_loss': 0, 'pot_loss': 0}

    union_feat_path = os.path.join(MODELS_DIR, "union_features_mother.pkl")
    if not os.path.exists(union_feat_path) or total_rows_eval == 0:
        return None, d

    current_feat_names = joblib.load(union_feat_path)
    current_model_top_k = len(current_feat_names) // TIME_STEPS

    all_preds_eval = np.zeros(total_rows_eval)
    available_cats_eval = []

    for c_id in ['0', '1']:
        mask = (cat_ids_eval == int(c_id))
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        m_path = os.path.join(CAT_MODELS_PATH, f"model_{c_id}.keras")
        s_path = os.path.join(SCALERS_PATH, f"scaler_{c_id}.pkl")
        if not (os.path.exists(m_path) and os.path.exists(s_path)):
            continue

        try:
            scaler = joblib.load(s_path)
            model = tf.keras.models.load_model(m_path, custom_objects={'MultiHeadSelfAttention': MultiHeadSelfAttention})

            X_cat_raw = df_eval.iloc[indices][current_feat_names]
            X_scaled = scaler.transform(X_cat_raw.values)
            X_3d = X_scaled.reshape(-1, current_model_top_k, TIME_STEPS)
            X_3d = np.transpose(X_3d, (0, 2, 1))

            raw_preds = model.predict(X_3d, batch_size=256, verbose=0).flatten()

            cal_path = os.path.join(MODELS_DIR, f"calibrator_{c_id}.pkl")
            if os.path.exists(cal_path):
                calibrator = joblib.load(cal_path)
                cat_preds = calibrator.predict_proba(raw_preds.reshape(-1, 1))[:, 1]
            else:
                cat_preds = raw_preds

            trend_probs_cat = predict_trend_probability(df_eval.iloc[indices])
            cat_preds, _ = combine_daughter_and_trend(cat_preds, trend_probs_cat, c_id)

            all_preds_eval[indices] = cat_preds
            available_cats_eval.append(c_id)
        except Exception as e:
            print(Fore.YELLOW + f"⚠️ [Shadow Holdout] خطا در پیش‌بینی دسته {c_id}: {e}. این دسته رد شد.")
            continue

    if len(available_cats_eval) == 0:
        return None, d

    for i in range(total_rows_eval):
        c_str = str(cat_ids_eval[i])
        if c_str not in available_cats_eval:
            continue

        y_val = y_real_eval[i]
        is_win = y_val > 0
        pred = all_preds_eval[i]
        th = best_thresholds.get(c_str, 0.5)

        d['base_tr'] += 1
        if is_win:
            d['base_wins'] += 1
            d['base_bal'] += d['base_bal'] * 0.01
        else:
            d['base_bal'] -= d['base_bal'] * 0.01
        if not is_win:
            d['pot_loss'] += 1

        if pred >= th:
            d['ai_tr'] += 1
            if is_win:
                d['ai_wins'] += 1
                d['ai_bal'] += d['ai_bal'] * 0.01
            else:
                d['ai_bal'] -= d['ai_bal'] * 0.01
        else:
            if not is_win:
                d['saved_loss'] += 1

        d['base_peak'] = max(d['base_peak'], d['base_bal'])
        d['base_mdd'] = max(d['base_mdd'], (d['base_peak'] - d['base_bal']) / d['base_peak'])
        d['ai_peak'] = max(d['ai_peak'], d['ai_bal'])
        d['ai_mdd'] = max(d['ai_mdd'], (d['ai_peak'] - d['ai_bal']) / d['ai_peak'])

    kpi_score = calculate_unified_kpi(d)
    return kpi_score, d


def log_shadow_holdout_result(trial, val_kpi, val_d, holdout_kpi, holdout_d):
    """
    🆕 [SHADOW EVAL] هسته‌ی کل قابلیت Shadow Holdout Evaluation.
    هر Trial، علاوه بر KPI روی Validation (که Optuna واقعاً می‌بیند و باهاش
    تصمیم می‌گیرد)، KPI روی Final-Test/Holdout هم silent محاسبه و در یک
    فایل CSV لاگ می‌شود - این عدد هرگز به Optuna داده نمی‌شود (یعنی امکان
    overfitting-via-selection روی Holdout صفر است؛ Holdout فقط برای
    *مشاهده*ی انسانی بعد از هر Trial استفاده می‌شود، نه برای تصمیم‌گیری).
    اگر Holdout KPI به‌طور محسوسی از Validation KPI عقب بماند، این نشانه‌ی
    کلاسیک overfitting-via-selection روی Validation است - یعنی هایپرپارامترهای
    این Trial با ویژگی‌های خاصِ همان نیمه از داده عجین شده‌اند و روی داده‌ی
    واقعاً ندیده به همان اندازه تعمیم پیدا نمی‌کنند.

    🆕 [OVERFIT DETECTOR] در انتهای همین تابع، analyze_shadow_holdout_trend()
    خودکار صدا زده می‌شود تا کل تاریخچه‌ی CSV (نه فقط همین ردیف) بررسی و
    وضعیت Overfitting/بهترین Trial واقعی/نقطه‌ی شروع Overfitting گزارش شود.
    """
    log_path = os.path.join(MODELS_DIR, "Shadow_Holdout_Log.csv")
    file_exists = os.path.exists(log_path)

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("trial_number,timestamp,val_kpi,holdout_kpi,gap,val_balance,holdout_balance,val_mdd,holdout_mdd,val_trades,holdout_trades\n")

            # 🐛 [FIX] قبلاً ردیف حالت "بدون Holdout" با شمارش دستی کاما نوشته
            # می‌شد (",,,,,,,\n") که یک کاما کم داشت و باعث می‌شد ستون‌های CSV
            # از ردیف بعد به بعد یک خانه جابه‌جا بخوانده شوند. حالا هر دو حالت
            # با ساختن یک لیست از ۱۱ مقدار (دقیقاً هم‌تعداد با هدر) و join با
            # کاما نوشته می‌شود - از نظر تعداد ستون تضمینی درست است.
            if holdout_kpi is not None:
                gap = val_kpi - holdout_kpi
                row_values = [
                    trial.number,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    f"{val_kpi:.6f}",
                    f"{holdout_kpi:.6f}",
                    f"{gap:.6f}",
                    f"{val_d.get('ai_bal', float('nan')):.2f}",
                    f"{holdout_d.get('ai_bal', float('nan')):.2f}",
                    f"{val_d.get('ai_mdd', float('nan')):.4f}",
                    f"{holdout_d.get('ai_mdd', float('nan')):.4f}",
                    f"{val_d.get('ai_tr', 0)}",
                    f"{holdout_d.get('ai_tr', 0)}",
                ]
            else:
                row_values = [
                    trial.number,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    f"{val_kpi:.6f}",
                    "", "", "", "", "", "", "", "",
                ]
            f.write(",".join(str(v) for v in row_values) + "\n")
    except Exception as e:
        print(Fore.RED + f"⚠️ خطا در نوشتن Shadow_Holdout_Log.csv: {e}. رد شد، بدون کرش.")

    if holdout_kpi is None:
        print(Fore.YELLOW + f"ℹ️ [Shadow Holdout] Trial {trial.number}: ارزیابی روی Holdout ممکن نشد (داده/مدل کافی نبود).")
    else:
        gap = val_kpi - holdout_kpi
        print(Fore.CYAN + f"👁️  [Shadow Holdout] Trial {trial.number}: Val KPI={val_kpi:.4f} | Holdout KPI={holdout_kpi:.4f} | Gap={gap:+.4f}")

        # 🚨 هشدار Overfitting فوری: اگر Holdout کمتر از نصف چیزی که Validation قول داده به دست آورد
        if val_kpi > 0 and holdout_kpi < val_kpi * 0.5:
            print(Fore.RED + f"🚨 [Shadow Holdout] هشدار Overfitting احتمالی در Trial {trial.number}: "
                              f"KPI روی Holdout ({holdout_kpi:.4f}) کمتر از نصف KPI روی Validation ({val_kpi:.4f}) است.")

    # 🆕 [OVERFIT DETECTOR] تحلیل کل تاریخچه (نه فقط این Trial) - همیشه در پایان صدا زده می‌شود
    try:
        analyze_shadow_holdout_trend()
    except Exception as e:
        print(Fore.RED + f"⚠️ خطا در analyze_shadow_holdout_trend: {e}. رد شد، بدون کرش و بدون تاثیر روی Trial.")


def analyze_shadow_holdout_trend():
    """
    🆕 [OVERFIT DETECTOR] هر بار که Shadow_Holdout_Log.csv آپدیت می‌شود (یعنی
    بعد از هر Trial)، این تابع صدا زده می‌شود و کل تاریخچه‌ی CSV (نه فقط
    ردیف تازه) را می‌خواند تا سه سوال را جواب دهد:

      ۱) آیا فرایند جست‌وجوی Optuna دارد روی Validation overfit می‌کند؟
         سنجه: همبستگی Pearson بین Val KPI و Holdout KPI (کلی + پنجره‌ی
         اخیر) و شیب رگرسیون خطی Gap (=Val-Holdout) نسبت به شماره‌ی Trial.
         اگر بهبود روی Validation واقعی باشد، باید همبستگی مثبت و قوی
         بماند و Gap رشد نکند؛ اگر جست‌وجو دارد نویز مخصوص Validation را
         حفظ می‌کند (overfitting-via-selection)، همبستگی افت می‌کند و/یا
         Gap با گذشت Trial ها بزرگ‌تر می‌شود.

      ۲) بهترین Trial از نظر عملکرد واقعی (Holdout) کدام است؟ ممکن است با
         «بهترین Trial از نظر Optuna» (که فقط Val را دیده) متفاوت باشد -
         این تفاوت خودش شاهد مستقیم overfitting-via-selection است.

      ۳) اگر overfitting در حال وقوع است، دقیقاً از کدام Trial به بعد شروع
         شده؟ با دنبال کردن دنباله‌ی «رکوردهای پیاپی روی Val» و مقایسه‌ی
         Holdout هم‌زمان هرکدام: تا جایی که هر رکورد جدید روی Val با بهتر
         شدن واقعی Holdout همراه است، سالم است. از جایی که رکوردهای جدید
         روی Val دیگر با بهتر شدن Holdout همراه نیستند (Holdout راکد/نزولی
         می‌ماند)، آن نقطه به‌عنوان شروع احتمالی overfitting علامت می‌خورد.

    خروجی‌ها (هر دو همیشه overwrite می‌شوند - رشد فایل با تعداد Trial صفر است):
      - Overfitting_Analysis_Summary.txt  : خلاصه‌ی متنی کامل
      - Overfitting_Analysis_Dashboard.jpg: داشبورد ۴ نموداره
    و یک خلاصه‌ی ۳-۴ خطی هم در کنسول چاپ می‌شود.
    """
    log_path = os.path.join(MODELS_DIR, "Shadow_Holdout_Log.csv")
    if not os.path.exists(log_path):
        return

    try:
        df_log = pd.read_csv(log_path)
    except Exception as e:
        print(Fore.RED + f"⚠️ [Overfitting Analysis] خطا در خواندن Shadow_Holdout_Log.csv: {e}")
        return

    if 'holdout_kpi' not in df_log.columns:
        return

    df_valid = df_log.dropna(subset=['holdout_kpi']).sort_values('trial_number').reset_index(drop=True)

    MIN_TRIALS_FOR_ANALYSIS = 5   # زیر این تعداد، آمار همبستگی/روند بی‌معنا و پرنویز است
    if len(df_valid) < MIN_TRIALS_FOR_ANALYSIS:
        print(Fore.WHITE + f"ℹ️ [Overfitting Analysis] هنوز فقط {len(df_valid)} Trial با Holdout معتبر داریم "
                            f"(حداقل {MIN_TRIALS_FOR_ANALYSIS} لازم است) - تحلیل صبر می‌کند.")
        return

    trials = df_valid['trial_number'].values.astype(float)
    val_k = df_valid['val_kpi'].values.astype(float)
    hold_k = df_valid['holdout_kpi'].values.astype(float)
    gap = val_k - hold_k

    # --- ۱. همبستگی کلی و همبستگی پنجره‌ی اخیر ---
    overall_corr = float(np.corrcoef(val_k, hold_k)[0, 1])
    recent_window = max(5, min(15, len(df_valid) // 2))
    recent_corr = float(np.corrcoef(val_k[-recent_window:], hold_k[-recent_window:])[0, 1])

    # --- ۲. شیب رگرسیون خطی Gap نسبت به شماره‌ی Trial (رشد Gap = بدتر شدن) ---
    gap_slope = float(np.polyfit(trials, gap, 1)[0])

    # --- ۳. بهترین Trial از نظر عملکرد واقعی (Holdout) در مقابل بهترین از نظر Optuna (Val) ---
    # 🐛 [FIX] دیگر صرفاً argmax(holdout_kpi) نیست: یک Trial با KPI بالا ولی
    # Drawdown فاجعه‌بار (مثلاً بالای MAX_ACCEPTABLE_HOLDOUT_MDD) از رقابت حذف
    # می‌شود، چون با آن‌قدر drawdown، جهت نهایی balance به‌شدت به توالی تصادفی
    # معاملات وابسته است، نه شاهدی از تعمیم واقعی مدل - همان فیلتری که در
    # get_true_best_trial_from_shadow_log هم اعمال شده، این‌جا هم تکرار
    # می‌شود تا داشبورد/خلاصه با منطق تصمیم‌گیری نهایی هماهنگ بماند.
    if 'holdout_mdd' in df_valid.columns:
        safe_mask = df_valid['holdout_mdd'].values.astype(float) <= MAX_ACCEPTABLE_HOLDOUT_MDD
    else:
        safe_mask = np.ones(len(df_valid), dtype=bool)

    if safe_mask.any():
        safe_hold_k = np.where(safe_mask, hold_k, -np.inf)
        best_holdout_idx = int(np.argmax(safe_hold_k))
        best_holdout_trial, best_holdout_val_kpi, best_holdout_kpi = int(trials[best_holdout_idx]), float(val_k[best_holdout_idx]), float(hold_k[best_holdout_idx])
    else:
        # هیچ Trial ای زیر سقف MDD نیست - fallback به argmax خام، ولی با هشدار صریح
        best_holdout_idx = int(np.argmax(hold_k))
        best_holdout_trial, best_holdout_val_kpi, best_holdout_kpi = int(trials[best_holdout_idx]), float(val_k[best_holdout_idx]), float(hold_k[best_holdout_idx])
        print(Fore.RED + f"   🚨 هشدار: هیچ Trial ای Holdout MDD کمتر از {MAX_ACCEPTABLE_HOLDOUT_MDD*100:.0f}٪ ندارد؛ "
                          f"«بهترین واقعی» نمایش داده‌شده (#{best_holdout_trial}) هم پرریسک است - برای Live مناسب نیست.")

    best_val_idx = int(np.argmax(val_k))
    best_val_trial, best_val_kpi_val, best_val_holdout_kpi = int(trials[best_val_idx]), float(val_k[best_val_idx]), float(hold_k[best_val_idx])

    # --- ۴. ردیابی رکوردهای پیاپی روی Val و Holdout هم‌زمانشان -> نقطه‌ی شروع Overfitting ---
    running_best_val = -float('inf')
    records = []  # (trial, val, holdout) - فقط لحظاتی که یک رکورد جدید روی Val ثبت شده
    for t, v, h in zip(trials, val_k, hold_k):
        if v > running_best_val:
            running_best_val = v
            records.append((int(t), float(v), float(h)))

    onset_trial = None
    onset_detail = ""
    if len(records) >= 3:
        peak_i = max(range(len(records)), key=lambda i: records[i][2])
        peak_trial, peak_val, peak_holdout = records[peak_i]
        later = records[peak_i + 1:]
        if len(later) >= 2 and peak_holdout > 0:
            later_avg_holdout = float(np.mean([r[2] for r in later]))
            if later_avg_holdout < peak_holdout * 0.7:
                onset_trial = peak_trial
                drop_pct = (1 - later_avg_holdout / peak_holdout) * 100
                onset_detail = (
                    f"تا Trial #{peak_trial}: هر بار Val رکورد زده، Holdout هم واقعاً بهتر شده "
                    f"(اوج Holdout همان‌جا: {peak_holdout:.4f}).\n"
                    f"بعد از آن {len(later)} رکورد جدید روی Val ثبت شده، ولی میانگین Holdout "
                    f"همراهشان فقط {later_avg_holdout:.4f} بوده ({drop_pct:.0f}% افت نسبت به اوج) -\n"
                    f"یعنی از Trial #{peak_trial} به بعد، «بهتر شدن روی Val» دیگر «بهتر شدن واقعی» را تضمین نمی‌کند."
                )

    # --- ۵. طبقه‌بندی نهایی وضعیت ---
    if recent_corr >= 0.5 and gap_slope <= 0:
        status = "✅ Healthy — بهبودها روی Validation واقعاً به Holdout منتقل می‌شوند"
        status_color = Fore.GREEN
    elif recent_corr >= 0.1:
        status = "⚠️ Warning — همبستگی ضعیف شده یا Gap در حال رشد است، مراقب باش"
        status_color = Fore.YELLOW
    else:
        status = "🚨 Overfitting Detected — بهتر شدن روی Validation دیگر با بهبود واقعی هم‌راستا نیست"
        status_color = Fore.RED

    # --- خروجی کنسول: خلاصه‌ی کوتاه (چون این تابع تا ۳۰۰ بار اجرا می‌شود، مختصر نگه داشته شده) ---
    print(status_color + f"📐 [Overfitting Analysis | {len(df_valid)} Trial] {status}")
    print(status_color + f"   همبستگی کلی: {overall_corr:.2f} | همبستگی {recent_window} Trial اخیر: {recent_corr:.2f} | شیب Gap: {gap_slope:+.5f}/Trial")
    print(status_color + f"   بهترین از نظر Optuna: Trial #{best_val_trial} (Val={best_val_kpi_val:.4f}, Holdout={best_val_holdout_kpi:.4f})  |  "
                          f"بهترین واقعی: Trial #{best_holdout_trial} (Val={best_holdout_val_kpi:.4f}, Holdout={best_holdout_kpi:.4f})")
    if onset_trial is not None:
        print(Fore.RED + f"   🚨 نقطه‌ی احتمالی شروع Overfitting: Trial #{onset_trial}")

    # --- فایل متنی خلاصه (همیشه overwrite - بدون رشد دیسک) ---
    summary_path = os.path.join(MODELS_DIR, "Overfitting_Analysis_Summary.txt")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("   Shadow Holdout Overfitting Analysis\n")
            f.write(f"   Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Trials analyzed (with valid Holdout eval): {len(df_valid)}\n\n")
            f.write(f"Overall correlation (Val vs Holdout)     : {overall_corr:.3f}\n")
            f.write(f"Recent correlation (last {recent_window} trials)     : {recent_corr:.3f}\n")
            f.write(f"Gap trend slope                          : {gap_slope:+.6f} per trial\n\n")
            f.write(f"STATUS: {status}\n\n")
            f.write("-" * 70 + "\n")
            f.write(f"Best trial by Optuna (Validation)  : #{best_val_trial}  | Val={best_val_kpi_val:.4f} | Holdout={best_val_holdout_kpi:.4f}\n")
            f.write(f"Best trial by TRUE generalization   : #{best_holdout_trial}  | Val={best_holdout_val_kpi:.4f} | Holdout={best_holdout_kpi:.4f}\n")
            f.write("-" * 70 + "\n\n")
            if onset_trial is not None:
                f.write(f"⚠️ Possible Overfitting Onset: Trial #{onset_trial}\n\n")
                f.write(onset_detail + "\n\n")
            else:
                f.write("No clear overfitting onset point detected yet (or not enough val-record events).\n\n")
            f.write("RECOMMENDATION:\n")
            if best_val_trial == best_holdout_trial:
                f.write(f"  Trial #{best_val_trial} is best by BOTH criteria - safe to trust as-is.\n")
            else:
                f.write(f"  Optuna's official best (#{best_val_trial}) differs from the true best (#{best_holdout_trial}).\n")
                f.write(f"  Consider manually inspecting Trial #{best_holdout_trial}'s params/backup folder\n")
                f.write(f"  as a more honest alternative to the final auto-retrain.\n")
            f.write("=" * 70 + "\n")
    except Exception as e:
        print(Fore.RED + f"⚠️ خطا در نوشتن Overfitting_Analysis_Summary.txt: {e}")

    # --- داشبورد تصویری ۴ نموداره (همیشه overwrite - بدون رشد دیسک) ---
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # (۱) روند Val/Holdout در طول Trial ها
        ax1 = axes[0, 0]
        ax1.plot(trials, val_k, marker='o', markersize=3, label='Validation KPI', color='#1e90ff', alpha=0.8)
        ax1.plot(trials, hold_k, marker='o', markersize=3, label='Holdout KPI', color='#ff6348', alpha=0.8)
        ax1.scatter([best_holdout_trial], [best_holdout_kpi], color='gold', s=150, zorder=5, edgecolor='black', label=f'بهترین واقعی (#{best_holdout_trial})')
        if onset_trial is not None:
            ax1.axvline(onset_trial, color='red', linestyle='--', alpha=0.6, label=f'شروع احتمالی Overfit (#{onset_trial})')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('KPI')
        ax1.set_title('Validation KPI در مقابل Holdout KPI')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.2)

        # (۲) اسکترپلات Val در مقابل Holdout
        ax2 = axes[0, 1]
        ax2.scatter(val_k, hold_k, color='#2ed573', alpha=0.6, edgecolor='black', linewidth=0.3)
        lims = [min(val_k.min(), hold_k.min()), max(val_k.max(), hold_k.max())]
        ax2.plot(lims, lims, linestyle='--', color='gray', label='y = x (تطابق کامل)')
        ax2.set_xlabel('Validation KPI')
        ax2.set_ylabel('Holdout KPI')
        ax2.set_title(f'همبستگی Val/Holdout (r = {overall_corr:.2f})')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.2)

        # (۳) روند Gap (Val - Holdout)
        ax3 = axes[1, 0]
        ax3.bar(trials, gap, color=['#ff4757' if g > 0 else '#2ed573' for g in gap], alpha=0.6)
        roll_window = max(3, len(df_valid) // 10)
        rolling_gap = pd.Series(gap).rolling(roll_window, min_periods=1).mean()
        ax3.plot(trials, rolling_gap, color='black', linewidth=2, label=f'میانگین متحرک ({roll_window})')
        ax3.axhline(0, color='gray', linewidth=0.8)
        ax3.set_xlabel('Trial Number')
        ax3.set_ylabel('Gap (Val - Holdout)')
        ax3.set_title('فاصله‌ی Val از واقعیت (قرمز = Val خوش‌بینانه‌تر از واقعیت)')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.2)

        # (۴) همبستگی متحرک در طول زمان
        ax4 = axes[1, 1]
        if len(df_valid) >= 6:
            corr_window = max(5, len(df_valid) // 6)
            rolling_corrs, rolling_trials = [], []
            for i in range(corr_window, len(df_valid) + 1):
                c = np.corrcoef(val_k[i - corr_window:i], hold_k[i - corr_window:i])[0, 1]
                rolling_corrs.append(c)
                rolling_trials.append(trials[i - 1])
            ax4.plot(rolling_trials, rolling_corrs, color='#8e44ad', linewidth=2)
            ax4.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='آستانه‌ی Healthy (0.5)')
            ax4.axhline(0.1, color='orange', linestyle='--', alpha=0.5, label='آستانه‌ی Warning (0.1)')
            ax4.axhline(0, color='gray', linewidth=0.5)
            ax4.set_ylim(-1.05, 1.05)
            ax4.set_xlabel('Trial Number')
            ax4.set_ylabel(f'همبستگی متحرک (پنجره={corr_window})')
            ax4.set_title('روند همبستگی Val/Holdout در طول زمان')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.2)
        else:
            ax4.text(0.5, 0.5, 'داده‌ی کافی برای\nهمبستگی متحرک نیست', ha='center', va='center', fontsize=11, transform=ax4.transAxes)
            ax4.set_xticks([])
            ax4.set_yticks([])

        plt.suptitle(f'Shadow Holdout Overfitting Dashboard — {status}', fontsize=13)
        plt.tight_layout()
        dashboard_path = os.path.join(MODELS_DIR, "Overfitting_Analysis_Dashboard.jpg")
        plt.savefig(dashboard_path, dpi=150)
        if run_optuna == 0:
            plt.show()
        else:
            plt.close('all')
    except Exception as e:
        print(Fore.RED + f"⚠️ خطا در رسم داشبورد Overfitting: {e}. رد شد، بدون کرش.")


def get_true_best_trial_from_shadow_log(min_trials=5, max_holdout_mdd=MAX_ACCEPTABLE_HOLDOUT_MDD):
    """
    🆕 [SHADOW EVAL] بهترین Trial را بر اساس عملکرد واقعی (Holdout KPI) از
    Shadow_Holdout_Log.csv استخراج می‌کند - نه بر اساس Validation KPI (که
    معیار رسمی خودِ Optuna است و ممکن است overfitting-via-selection داشته
    باشد). این تابع در دو جا استفاده می‌شود:
      1) داخل analyze_shadow_holdout_trend برای گزارش‌های میانی هر Trial
      2) در Final Retrain (پایان اجرای Optuna) برای تصمیم‌گیری این‌که مدل
         نهایی که در MODELS_DIR ذخیره می‌شود، بر اساس کدام Trial باشد.

    🐛 [FIX] نسخه‌ی قبلی این تابع فقط argmax(holdout_kpi) را برمی‌گرداند -
    یعنی یک Trial با KPI بالا ولی Drawdown فاجعه‌بار (مثلاً ۷۰-۸۵٪) هم می‌توانست
    به‌عنوان «بهترین» انتخاب شود، در حالی که با تعداد معاملات زیاد (پس نه
    نمونه‌ی کوچک) و MDD این‌قدر بزرگ، نتیجه‌ی نهایی به‌شدت به توالی
    (sequence) معاملات وابسته است - یعنی جهت نهایی balance عملاً شانسی است،
    نه شاهدی از تعمیم واقعی. حالا فقط Trial هایی که holdout_mdd شان کمتر یا
    مساوی max_holdout_mdd باشد وارد رقابت می‌شوند؛ اگر هیچ‌کدام این شرط را
    نداشته باشند، (None, None, None) برمی‌گردد (یعنی «هنوز کاندید امنی پیدا
    نشده»)، به‌جای انتخاب کورکورانه‌ی پرریسک‌ترین برنده.

    خروجی: (trial_number, holdout_kpi, val_kpi) یا (None, None, None) اگر
    داده‌ی کافی/کاندید امن نباشد.
    """
    log_path = os.path.join(MODELS_DIR, "Shadow_Holdout_Log.csv")
    if not os.path.exists(log_path):
        return None, None, None

    try:
        df_log = pd.read_csv(log_path)
        df_valid = df_log.dropna(subset=['holdout_kpi'])
        if len(df_valid) < min_trials:
            return None, None, None

        if 'holdout_mdd' in df_valid.columns:
            df_safe = df_valid[df_valid['holdout_mdd'] <= max_holdout_mdd]
        else:
            df_safe = df_valid  # سازگاری با لاگ‌های خیلی قدیمی که هنوز ستون MDD نداشتند

        if len(df_safe) == 0:
            print(Fore.YELLOW + f"⚠️ [Shadow Holdout] هیچ Trial ای با Holdout MDD زیر {max_holdout_mdd*100:.0f}٪ پیدا نشد "
                                 f"- همه‌ی کاندیدها ریسک بیش از حد دارند، انتخاب امنی وجود ندارد.")
            return None, None, None

        best_idx = df_safe['holdout_kpi'].idxmax()
        best_row = df_safe.loc[best_idx]
        return int(best_row['trial_number']), float(best_row['holdout_kpi']), float(best_row['val_kpi'])
    except Exception as e:
        print(Fore.RED + f"⚠️ خطا در خواندن بهترین Trial واقعی از Shadow_Holdout_Log.csv: {e}")
        return None, None, None


# ================= Optuna Setup =================
print("\n6. Optuna Optimization (Auto-Tuning):")
run_optuna = int(input("   Run Optuna? (1 = Yes, 0 = No) : "))



# فقط یک مقداردهی اولیه ساده انجام می‌دهیم. 
# مقدار اصلی بعداً در بخش اجرای Optuna از دیتابیس خوانده می‌شود.
global glob_best_kpi
glob_best_kpi = -999999.0




# 🆕 تقسیم سه‌تایی Train / Validation / Final-Test
# -----------------------------------------------------------------------
# Train      = DATASET_PATH (بدون تغییر - همون‌جایی که مدل روش فیت می‌شود)
# Validation = نیمی از ردیف‌های TEST_DATA_PATH (یکی‌درمیون) - همینه که هر
#              Trial از Optuna (۳۰۰ بار) روش KPI می‌سنجه و باهاش هایپرپارامتر
#              انتخاب می‌کند. طبیعیه که هایپرپارامترهای نهایی کمی با این
#              نیمه عجین بشن (پدیده‌ی معروف به overfitting-via-selection).
# Final-Test = نیم دیگه‌ی TEST_DATA_PATH (یکی‌درمیون، ردیف‌های باقی‌مانده) -
#              کاملاً دست‌نخورده می‌ماند و فقط یک‌بار، در بازآموزی نهایی بعد
#              از قفل‌شدن بهترین هایپرپارامترها، برای گزارش صادقانه‌ی
#              عملکرد واقعی استفاده می‌شود.
# این پرچم پایین‌تر (درست قبل از بازآموزی نهایی) True می‌شود.
USE_FINAL_HOLDOUT_TEST = False





def _cleanup_old_backups(desktop_path, prefix, keep):
    """پاک کردن قدیمی‌ترین بکاپ‌ها وقتی تعدادشان از `keep` بیشتر شد (فقط برای پوشه‌هایی که با `prefix` شروع می‌شوند)."""
    try:
        candidates = [
            os.path.join(desktop_path, name)
            for name in os.listdir(desktop_path)
            if name.startswith(prefix) and os.path.isdir(os.path.join(desktop_path, name))
        ]
        candidates.sort(key=lambda p: os.path.getctime(p))  # قدیمی‌ترین اول

        excess = len(candidates) - keep
        if excess > 0:
            for old_folder in candidates[:excess]:
                shutil.rmtree(old_folder, ignore_errors=True)
                print(Fore.YELLOW + f"🧹 بکاپ قدیمی پاک شد (برای نگه‌داشتن فقط {keep} تای آخر): {os.path.basename(old_folder)}")
    except Exception as e:
        print(Fore.RED + f"⚠️ خطا در پاک‌سازی بکاپ‌های قدیمی: {e}")


def backup_best_models_folder(kpi_score, label):
    """
    🆕 از کل پوشه‌ی MODELS_DIR (مدل‌های Keras، اسکیلرها، کالیبراتورها، کش‌ها،
    optuna_memory.db و ...) یک کپی کامل روی دسکتاپ گرفته می‌شود.

    دو حالت استفاده می‌شود:
      1) label="BEST_Trial{شماره}"  → هر بار Optuna طول جستجو رکورد جدید KPI
         (روی Validation) پیدا کند. این نوع بکاپ‌ها محدود به آخرین
         MAX_BACKUPS_TO_KEEP تا نگه داشته می‌شوند (rotation خودکار).
      2) label="FINAL_HOLDOUT"      → فقط یک‌بار در پایان، برای نتیجه‌ی
         نهایی و صادقانه روی داده‌ی کاملاً دست‌نخورده. این بکاپ هیچ‌وقت
         خودکار پاک نمی‌شود.

    اسم پوشه‌ی بکاپ شامل: label + مقدار KPI + تاریخ و ساعت است.
    """
    try:
        # MODELS_DIR از قبل داخل دسکتاپ است، پس پوشه‌ی پدرش همان دسکتاپ کاربر است
        desktop_path = os.path.dirname(MODELS_DIR)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        backup_folder_name = f"{label}_KPI_{kpi_score:.4f}_{timestamp}"
        backup_full_path = os.path.join(desktop_path, backup_folder_name)

        shutil.copytree(MODELS_DIR, backup_full_path, dirs_exist_ok=True)
        print(Fore.GREEN + f"💾 بکاپ کامل گرفته شد:\n   {backup_full_path}")

        if label.startswith("BEST_Trial"):
            _cleanup_old_backups(desktop_path, prefix="BEST_Trial", keep=MAX_BACKUPS_TO_KEEP)
    except Exception as e:
        # یک خطای بکاپ نباید کل فرایند بهینه‌سازی ۳۰۰ ترایالی را متوقف کند
        print(Fore.RED + f"⚠️ خطا در گرفتن بکاپ از پوشه‌ی مدل‌ها: {e}")


def _apply_optuna_trial_suggestions(trial):
    """
    🆕 [FIX #4] فقط زمانی صدا زده می‌شود که یک Trial واقعی از Optuna در جریان
    باشد (یعنی objective() با trial != None فراخوانی شده). تمام
    trial.suggest_* و اجبار فازها (train_mother=1 و ...) اینجا متمرکز شده تا
    در حالت اجرای دستی طبق منو (trial=None) هیچ‌کدام اجرا نشوند و انتخاب‌های
    کاربر در «SYSTEM CONFIGURATION» دست‌نخورده باقی بمانند.
    """
    global epoch_BASE, base_patience, base_learning_rate, class_weights_win_Mother
    global base_GaussianNoise, epoch_cat, cat_learning_rate, DAUGHTER_WEIGHTS_CONFIG, cat_patience
    global X_layeAkhar_UnFreez, TOP_K, N1_Dropout, N2_Dropout, N3_Dropout
    global train_mother, train_daughter, th_choice, run_test
    global base_l2_reg

    # --- 1. بهینه‌سازی مدل مادر ---
    # ایپاک را بالا می‌گیریم تا EarlyStopping تصمیم‌گیرنده اصلی باشد
    epoch_BASE = trial.suggest_int('epoch_BASE', 20, 100)
    base_patience = trial.suggest_int('base_patience', 4, 12)
    base_learning_rate = trial.suggest_float('base_learning_rate', 1e-5, 1e-3, log=True)

    # 🆕 بهینه‌سازی L2 به صورت لاگاریتمی (بسیار مهم در کنار Dropout)
    base_l2_reg = trial.suggest_float('base_l2_reg', 1e-4, 5e-2, log=True)

    class_weights_win_Mother = trial.suggest_float('class_weights_win_Mother', 1.0, 3.0)

    # نویز گوسی بهتر است خیلی شدید نباشد تا دیتای مالی تخریب نشود
    base_GaussianNoise = trial.suggest_float('base_GaussianNoise', 0.0005, 0.05)

    # --- 2. بهینه‌سازی مدل دختر ---
    # حداقل ایپاک را 5 می‌گذاریم تا وزن‌های GlorotUniform فرصت همگرایی داشته باشند
    epoch_cat = trial.suggest_int('epoch_cat', 5, 40)
    cat_patience = trial.suggest_int('cat_patience', 3, 10)

    # نرخ یادگیری دختر (Fine-tuning) معمولاً کمتر از مادر است
    cat_learning_rate = trial.suggest_float('cat_learning_rate', 1e-6, 3e-4, log=True)

    # وزن‌ها را معقول‌تر می‌کنیم تا مدل دچار بایاس شدید نشود
    DAUGHTER_WEIGHTS_CONFIG['0'] = trial.suggest_float('weight_sell_cat0', 0.8, 3.0)
    DAUGHTER_WEIGHTS_CONFIG['1'] = trial.suggest_float('weight_buy_cat1', 0.8, 3.0)

    X_layeAkhar_UnFreez = trial.suggest_int('X_layeAkhar_UnFreez', 1, 5)

    # --- 3. بهینه‌سازی ساختار و فیچرها ---
    # 🆕 [FIX #3] کف را از 10 به 5 کاهش دادیم: get_top_k_features حالا قبل از
    # mRMR یک تست معناداری آماری (permutation) روی خانواده‌ها اجرا می‌کند. اگر
    # در رژیم فعلی بازار فقط چند خانواده واقعاً سیگنال معنادار داشته باشند،
    # Optuna دیگر مجبور نیست حتماً چند خانواده‌ی نویزی اضافه را هم قاطی کند
    # تا به حداقل قدیمی (10) برسد؛ خودِ فیلتر معناداری همچنان مستقل از این
    # عدد، سقف واقعی تعداد خانواده‌های قابل‌استفاده را تعیین می‌کند.
    TOP_K = trial.suggest_int('TOP_K', 5, 28)

    # سقف Dropout را روی 0.5 می‌بندیم تا Underfitting رخ ندهد
    N1_Dropout = trial.suggest_float('N1_Dropout', 0.1, 0.5)
    N2_Dropout = trial.suggest_float('N2_Dropout', 0.1, 0.5)
    N3_Dropout = trial.suggest_float('N3_Dropout', 0.1, 0.5)

    # اجبار سیستم به اجرای کامل همه فازها - فقط در حین بهینه‌سازی Optuna
    train_mother = 1
    train_daughter = 2
    th_choice = 2
    run_test = 1


def _record_trial_result(trial, optuna_return_score, d_combined):
    """
    🆕 [FIX #4] ثبت نتیجه‌ی این اجرا. اگر trial واقعی از Optuna باشد، دقیقاً
    همان بوکیپینگ قبلی (مقایسه با glob_best_kpi، فایل‌های
    OPTUNA_LIVE_PROGRESS.txt / FINAL_HOLDOUT_RESULT.txt، بکاپ خودکار) بدون
    هیچ تغییری اجرا می‌شود. اگر trial=None باشد (اجرای دستی طبق منو)، مقایسه‌ی
    «بهترین در میان صدها Trial» اصلاً معنا ندارد (فقط یک اجرا داریم)، پس فقط
    KPI نهایی در کنسول چاپ می‌شود - گزارش کامل (Alpha_Performance_Report.txt یا
    FINAL_HOLDOUT_UNBIASED_Report.txt) چند خط بالاتر در همین فاز ۴ از قبل
    ذخیره شده و چیزی از دست نمی‌رود.
    """

    
    global glob_best_kpi

    try:
        if trial is None:
            print(Fore.CYAN + f"\n📊 KPI این اجرای دستی (Manual Run): {optuna_return_score:.6f}")
            return

        if USE_FINAL_HOLDOUT_TEST:
            # 🔒 این اجرا، بازآموزی نهایی روی Final-Test دست‌نخورده‌ست.
            # این عدد با KPI‌های روی Validation قابل مقایسه نیست (چون
            # روی دیتای متفاوتی حساب شده)، پس وارد بوکیپینگ glob_best_kpi
            # نمی‌شود؛ فقط یک گزارش و بکاپ جدا و صریح برایش ساخته می‌شود.
            holdout_file = os.path.join(MODELS_DIR, "FINAL_HOLDOUT_RESULT.txt")
            with open(holdout_file, "w", encoding="utf-8") as f:
                f.write("============================================================\n")
                f.write("   🔒 FINAL UNBIASED RESULT ON NEVER-SEEN HOLDOUT DATA\n")
                f.write(f"   Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("============================================================\n\n")

                f.write(f"📈 Final Holdout KPI Score: {optuna_return_score:.6f}\n")
                f.write(f"💰 Final Balance : {d_combined['ai_bal']:.2f}$\n")
                f.write(f"📉 Max Drawdown  : {d_combined['ai_mdd']*100:.2f}%\n\n")

                f.write("--- Hyperparameters Used (locked, from Optuna search) ---\n")
                f.write("------------------------------------------------------------\n")

                for param_name, param_value in trial.params.items():
                    f.write(f"👉 {param_name:<25} : {param_value}\n")

                f.write("\n============================================================")

            print(Fore.MAGENTA + "🔒 نتیجه‌ی نهایی و صادقانه روی داده‌ی کاملاً دست‌نخورده ذخیره شد: FINAL_HOLDOUT_RESULT.txt")
            print(Fore.MAGENTA + f"   KPI واقعی روی Holdout: {optuna_return_score:.6f}")

            # 🆕 بکاپ کامل و جدا از مدل‌های نهایی (این بکاپ هیچ‌وقت خودکار پاک نمی‌شود)
            backup_best_models_folder(optuna_return_score, label="FINAL_HOLDOUT")

        else:
            if optuna_return_score > glob_best_kpi:
                glob_best_kpi = optuna_return_score
                is_best = True
            else:
                is_best = False

            if is_best:
                current_best_file = os.path.join(MODELS_DIR, "OPTUNA_LIVE_PROGRESS.txt")
                with open(current_best_file, "w", encoding="utf-8") as f:
                    f.write("============================================================\n")
                    f.write("   🏆 NEW BEST PARAMETERS FOUND BY GEMINI ALPHA\n")
                    f.write(f"   Trial Number: {trial.number}\n")
                    f.write(f"   Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("============================================================\n\n")

                    f.write(f"📈 Best Optuna Score (Validation): {optuna_return_score:.6f}\n")
                    f.write(f"💰 Final Balance : {d_combined['ai_bal']:.2f}$\n")
                    f.write(f"📉 Max Drawdown  : {d_combined['ai_mdd']*100:.2f}%\n\n")

                    f.write("--- Optimized Hyperparameters List ---\n")
                    f.write("------------------------------------------------------------\n")

                    for param_name, param_value in trial.params.items():
                        f.write(f"👉 {param_name:<25} : {param_value}\n")

                    f.write("\n============================================================")

                print(Fore.YELLOW + f"✨ [Trial {trial.number}] Best Params Updated in File! --- ")

                # 🆕 بکاپ کامل از پوشه‌ی مدل‌ها روی دسکتاپ، هر بار که رکورد جدید KPI (روی Validation) ثبت می‌شود
                backup_best_models_folder(optuna_return_score, label=f"BEST_Trial{trial.number}")
    except Exception as e:
        print(Fore.RED + f"⚠️ Error updating progress file: {str(e)}")


def objective(trial=None):
    # فراخوانی متغیرهای شما برای تغییر داینامیک در طول بهینه‌سازی
    global epoch_BASE, base_patience, base_learning_rate, class_weights_win_Mother
    global base_GaussianNoise, epoch_cat, cat_learning_rate, DAUGHTER_WEIGHTS_CONFIG, cat_patience
    global X_layeAkhar_UnFreez, TOP_K, N1_Dropout, N2_Dropout, N3_Dropout
    global train_mother, train_daughter, th_choice, run_test
    
    # متغیر جدید برای L2 (باید در معماری مدل نیز اعمال شود)
    global base_l2_reg 

    tf.keras.backend.clear_session()

    if trial is not None:
        # 🆕 [FIX #4] پارامترها با trial.suggest_* تعیین می‌شوند و فازها اجباراً
        # کامل اجرا می‌گردند - فقط وقتی این تابع توسط Optuna (study.optimize)
        # فراخوانی شده باشد.
        _apply_optuna_trial_suggestions(trial)
        print(Fore.YELLOW + f"\n\n🚀 --- STARTING OPTUNA TRIAL {trial.number} --- 🚀")
    else:
        # 🆕 [FIX #4] اجرای دستی (بدون Optuna): هیچ overrideای روی هایپرپارامترها
        # یا روی انتخاب‌های منو (train_mother/train_daughter/th_choice/run_test)
        # انجام نمی‌شود - دقیقاً همان چیزی که در «SYSTEM CONFIGURATION» انتخاب
        # کرده‌ای اجرا می‌شود؛ هایپرپارامترها هم همان مقادیر پیش‌فرض بالای فایل‌اند.
        print(Fore.YELLOW + "\n\n▶️ --- MANUAL RUN (طبق انتخاب‌های منو، بدون Optuna) --- ▶️")
        print(Fore.WHITE + f"   Mother={train_mother} | Daughter={train_daughter} | Threshold={th_choice} | Backtest={run_test}")




# ------------------------------------------------------------------------------------------------------------------------------  PHASE 1 & 2





    # اجرای لود دیتا فقط در صورتی که نیاز به آموزش مدل مادر یا دختر باشد
    if train_mother == 1 or train_daughter in [0, 1, 2]:



        print(Fore.MAGENTA + "\n=== PHASE 1 & 2: Data Preprocessing & Training ===")
        


        # 1. لود کل دیتا (یک‌بار - قبلاً این خط به‌اشتباه دو بار پشت‌سرهم فراخوانی می‌شد)
        keys, X_raw_full, y = load_and_preprocess(DATASET_PATH)

        # === CALIB_SIZE به‌صورت پویا (درصدی) محاسبه می‌شود، نه عدد ثابت ===
        CALIB_SIZE = max(2000, int(len(y) * 0.10))
        print(Fore.CYAN + f"   ℹ️ CALIB_SIZE محاسبه‌شده: {CALIB_SIZE} سطر ({CALIB_SIZE/len(y)*100:.1f}% از {len(y)} سطر کل)")

        
        # 2. استخراج فیچرهای برتر (بر اساس mRMR/MI) برای کل Trial فعلی
        print(Fore.MAGENTA + f"--- Selecting TOP_{TOP_K} Features for Mother & Daughters (Leak-Safe) ---")
        union_features = get_top_k_features(DATASET_PATH, k_families=TOP_K, calib_size=CALIB_SIZE)
        
        # 🆕 استراتژی اجباری: استخراج و ضمیمه کردن (Append) ۱۰۰٪ تضمینی فیچرهای روند به فیچرهای برتر
        trend_features = get_trend_features(DATASET_PATH)
        if len(trend_features) > 0:
            # اضافه کردن فیچرهای روند بدون تکرار احتمالی
            additional_trend = [c for c in trend_features if c not in union_features]
            union_features = union_features + additional_trend
            print(Fore.GREEN + f"✅ تعداد {len(additional_trend)} فیچر روند ({len(additional_trend)//TIME_STEPS} خانواده) به‌صورت تضمینی به فیچرهای ورودی مدل اصلی ضمیمه شد.")
        else:
            print(Fore.YELLOW + "⚠️ هیچ ستون روندی در دیتاست پیدا نشد؛ مدل‌های اصلی فقط با Top-K ادامه می‌دهند.")

        # ذخیره لیست جامع فیچرها (Top-K + Trend) برای استفاده یکپارچه در تمام فازها
        joblib.dump(union_features, os.path.join(MODELS_DIR, "union_features_mother.pkl"))
        joblib.dump(union_features, os.path.join(MODELS_DIR, "selected_features_0.pkl"))
        joblib.dump(union_features, os.path.join(MODELS_DIR, "selected_features_1.pkl"))
        
        # فیلتر کردن دیتای خام با فیچرهای انتخابی جدید
        X_raw = X_raw_full[union_features]
        
        # محاسبه تعداد خانواده‌های واقعی در این Trial برای ورودی مدل
        current_k_families = len(union_features) // TIME_STEPS
        
        print(Fore.GREEN + f"✅ All models will use {len(union_features)} features ({current_k_families} families).")
        print(Fore.YELLOW + f"📉 دیتا با موفقیت فشرده شد: {X_raw.shape}")

        # =========================================================================
        # --- مدل روند (Trend Model): کاملاً مستقل از فیچرهای اصلی معاملاتی است ---
        # این مدل را قبل از حلقه‌ی مدل‌های دختر آموزش می‌دهیم تا خروجی OOF آن
        # (oof_trend_full) برای آموزش متا-مدل هر دسته (SELL/BUY) در دسترس باشد.
        # اگر ستون‌های روند در دیتاست موجود نباشند (مثلاً هنوز EA آپدیت نشده)،
        # این تابع None برمی‌گرداند و بقیه‌ی سیستم دقیقاً مثل قبل (بدون متا-مدل)
        # کار می‌کند - بدون کرش.
        # =========================================================================
        print(Fore.MAGENTA + "\n--- Step 0 (New): Training Trend Model (H4 Indicators) ---")
        oof_trend_full = prepare_trend_oof_for_training(DATASET_PATH, CALIB_SIZE)
        if oof_trend_full is None:
            print(Fore.YELLOW + "⚠️ مدل روند آموزش داده نشد. متا-مدل‌ها هم ساخته نمی‌شوند و سیستم فقط با مدل‌های دختر ادامه می‌دهد.")

        if X_raw is not None:
            keys_train = keys.iloc[:-CALIB_SIZE]
            X_raw_train = X_raw.iloc[:-CALIB_SIZE]
            y_train = y.iloc[:-CALIB_SIZE]

            mother_scaler = RobustScaler()
            mother_scaler.fit(X_raw_train)
            X_raw_for_mother = pd.DataFrame(
                mother_scaler.transform(X_raw), columns=X_raw.columns, index=X_raw.index
            )
            joblib.dump(mother_scaler, os.path.join(SCALERS_PATH, "scaler_mother.pkl"))

            X_3d = reshape_data_to_series(X_raw_for_mother)
            X_3d_train = X_3d[:-CALIB_SIZE]
            # ============================================

            if X_3d is not None:
                print(Fore.GREEN + f"✅ Reshape Successful: {X_3d_train.shape} (Train Set)")
                
                # --- PHASE 1: Mother Model ---
                
                if train_mother == 1:
                    print(Fore.MAGENTA + "\n--- Step 1: Training Base Model with Recursive K-Fold ---")   
                                    
                    tscv = TimeSeriesSplit(n_splits=Base_n_splits)
                    
                    # -------------- تعریف معماری   --------------  ( Base model )
                    inputs = Input(shape=( TIME_STEPS, current_k_families ))

                    # اضافه کردن نویز اولیه برای افزایش تاب‌آوری مدل در لایو
                    x_start = GaussianNoise(base_GaussianNoise)(inputs)

                    # ۱. استخراج ویژگی‌های محلی با دید وسیع (Dilation)
                    x_conv = Conv1D(N1_Conv1D, kernel_size=3, padding='same', activation='relu', dilation_rate=2)(x_start)
                    x_conv = BatchNormalization()(x_conv)

                    # ۲. تحلیل توالی و روند (LSTM)
                    x_lstm = Bidirectional(LSTM(N2_Bidirectional, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(x_conv)

                    # ۳. ایجاد میان‌بر (Shortcut) برای حفظ سیگنال‌های اولیه
                    shortcut = Conv1D(N2_Bidirectional * 2, kernel_size=1, padding='same')(x_conv)

                    # ۴. اتصال باقی‌مانده (Residual Connection)
                    x_res = tf.keras.layers.Add()([x_lstm, shortcut])
                    x_res = tf.keras.layers.Activation('relu')(x_res)

                    # ۵. تمرکز هوشمند روی گام‌های زمانی حساس (Attention)
                    x_att = MultiHeadSelfAttention(embed_dim=N2_Bidirectional * 2, num_heads=8)(x_res)
                    x_att = SpatialDropout1D(N2_Dropout)(x_att)

                    # ۶. فشرده‌سازی و لایه‌های تصمیم‌گیر نهایی
                    x_pool = GlobalAveragePooling1D()(x_att)
                    x_dense = Dense(N4_Dense, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(base_l2_reg))(x_pool)
                    x_dense = Dropout(N3_Dropout)(x_dense)
                    x_dense = BatchNormalization()(x_dense)

                    outputs = Dense(1, activation='sigmoid')(x_dense)

                    # ساخت و کامپایل مدل نهایی
                    base_model = Model(inputs, outputs)
                    base_model.compile(
                        optimizer=Adam(learning_rate=base_learning_rate), 
                        loss=BinaryCrossentropy(label_smoothing=0.1), 
                        metrics=['accuracy']
                    )

                    # صدا زدن تابع رسم برای این مدل
                    visualize_and_save_model(base_model, model_name="base_model", save_path=MODELS_DIR)


                                        




                    tscv = TimeSeriesSplit(n_splits=Base_n_splits)
                    
                    fold_no = 1
                    best_val_loss = float('inf')
                    fold_losses = []

                    all_train_loss = []
                    all_val_loss = []
                    all_train_acc = []
                    all_val_acc = []

                    


                    for train_index, val_index in tscv.split(X_3d_train):
                        print(Fore.CYAN + f"\n⏳ Training on Fold {fold_no}/{Base_n_splits}...")
                        
                        X_train_fold = X_3d_train[train_index]
                        y_train_fold = y_train.iloc[train_index]

                        future_start_idx = val_index[0]
                        X_future_val = X_3d_train[future_start_idx:]
                        y_future_val = y_train.iloc[future_start_idx:]

                        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_fold), y=y_train_fold)
                        class_weights_fold = {0: weights[0], 1: weights[1] * class_weights_win_Mother}

                        early_stop = EarlyStopping(monitor='val_loss', patience=base_patience, restore_best_weights=True)

                        history = base_model.fit(
                            X_train_fold, y_train_fold,
                            validation_data=(X_future_val, y_future_val),
                            epochs=epoch_BASE,
                            batch_size=32,
                            class_weight=class_weights_fold,
                            callbacks=[early_stop],
                            verbose=1
                        )


                        
                        
                        
                        all_train_loss.extend(history.history['loss'])
                        all_val_loss.extend(history.history['val_loss'])
                        all_train_acc.extend(history.history['accuracy'])
                        all_val_acc.extend(history.history['val_accuracy'])

                        

                        current_val_loss = min(history.history['val_loss'])
                        fold_losses.append(current_val_loss)
                        print(f"✅ Fold {fold_no} Completed. Val Loss: {current_val_loss:.4f}")                    
                        fold_no += 1

                    base_model.save(os.path.join(BASE_MODEL_PATH, "mother_model.keras"))
                    print(Fore.GREEN + "⭐ Best Mother Model Saved.")


                    
                    # --- رسم نمودار پیوسته برای تمام فولدها ---
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

                    ax1.plot(all_train_loss, label='Total Train Loss', color='blue')
                    ax1.plot(all_val_loss, label='Total Val Loss', color='orange')
                    
                    
                    ax1.set_xlabel('Total Epochs')
                    ax1.legend()

                    ax2.plot(all_train_acc, label='Total Train Accuracy', color='green')
                    ax2.plot(all_val_acc, label='Total Val Accuracy', color='red')
                    
                    
                    ax2.set_xlabel('Total Epochs')
                    ax2.legend()

                    plt.tight_layout()
                    plt.savefig(os.path.join(MODELS_DIR, "mother_performance_chart_continuous.jpg"))
                    plt.close()
                    
                    


                else:
                    print(Fore.WHITE + "\n--- Step 1: Base Model loaded from disk (Skipping training) ---")










                # --- PHASE 2: Daughter Models ----------------------------------------------------------------------------------------------  phase 2



                if train_daughter in [0, 1, 2]:
                    print(Fore.MAGENTA + "\n--- Step 2: Training Daughter Models with Recursive K-Fold ---")
                    df_full = pd.concat([keys_train, X_raw_train, y_train], axis=1)
                    groups = df_full.groupby(['ShortOrLong'])



                    for i, (name, group) in enumerate(groups):
                        current_name = name[0] if isinstance(name, tuple) else name
                        cat_id = str(int(current_name)) 
                        
                        if train_daughter == 0 and cat_id != '0': continue
                        if train_daughter == 1 and cat_id != '1': continue

                        print(Fore.CYAN + f"\n🚀 Starting Recursive Training for Category: {cat_id}")
                        
                        feat_path = os.path.join(MODELS_DIR, f"selected_features_{cat_id}.pkl")

                        selected_features_names = union_features
                        print(Fore.GREEN + f"✅ Daughter {cat_id} using same {len(selected_features_names)} features as Mother.")

                        # موقعیت‌های مطلق (نسبت به کل بازه‌ی train-only) مربوط به این دسته
                        # -> برای هم‌ترازی با oof_trend_full در ساخت متا-مدل لازم است
                        group_abs_positions = group.index.values

                        X_cat_raw = group[selected_features_names]
                        y_cat = group.iloc[:, -1].values
                        
                        scaler = StandardScaler()
                        X_cat_scaled = scaler.fit_transform(X_cat_raw)
                        joblib.dump(scaler, os.path.join(SCALERS_PATH, f"scaler_{cat_id}.pkl"))
                        
                        X_cat_3d = reshape_data_to_series(pd.DataFrame(X_cat_scaled))
                        
                        tf.keras.backend.clear_session()
                        fresh_base_model = tf.keras.models.load_model(
                            os.path.join(BASE_MODEL_PATH, "mother_model.keras"),
                            custom_objects={'MultiHeadSelfAttention': MultiHeadSelfAttention}
                        )
                        
                        daughter_model = build_transfer_model(fresh_base_model)
                        daughter_model.compile(
                            optimizer=Adam(learning_rate=cat_learning_rate), 
                            loss=BinaryCrossentropy(label_smoothing=0.15),
                            metrics=['accuracy']
                        ) 


                        # صدا زدن تابع رسم برای این مدل
                        visualize_and_save_model(daughter_model, model_name="daughter_model", save_path=MODELS_DIR)



                        tscv_daughter = TimeSeriesSplit(n_splits=daughter_n_splits)
                        fold_idx = 1
                        
                        current_gain_weight = DAUGHTER_WEIGHTS_CONFIG.get(cat_id, 1.0)
                        daughter_weights_dict = {0: 1.0, 1: float(current_gain_weight)}
                        print(f"   ⚖️ Applied Weights for Cat {cat_id}: {daughter_weights_dict}")
                        

                        all_train_loss = []
                        all_val_loss = []
                        all_train_acc = []
                        all_val_acc = []

                        # --- آرایه‌ی OOF مدل دختر برای این دسته (برای متا-مدل) ---
                        oof_daughter_preds = np.full(len(y_cat), np.nan, dtype='float32')


                        for train_idx, val_idx in tscv_daughter.split(X_cat_3d):
                            print(f"   ∟ 📂 Training Fold {fold_idx}/{daughter_n_splits} (Sequential)...")
                            
                            X_train_f = X_cat_3d[train_idx]
                            y_train_f = y_cat[train_idx]

                            future_start_cat = val_idx[0]
                            X_future_val_cat = X_cat_3d[future_start_cat:]
                            y_future_val_cat = y_cat[future_start_cat:]

                            early_stop = EarlyStopping(monitor='val_loss', patience=cat_patience, restore_best_weights=True)
                            
                            history_fold = daughter_model.fit(
                                X_train_f, y_train_f,
                                validation_data=(X_future_val_cat, y_future_val_cat),
                                epochs=epoch_cat,
                                batch_size=32,
                                class_weight=daughter_weights_dict,
                                verbose=1,
                                callbacks=[early_stop]
                            )
                            

                            

                            all_train_loss.extend(history_fold.history['loss'])
                            all_val_loss.extend(history_fold.history['val_loss'])
                            all_train_acc.extend(history_fold.history['accuracy'])
                            all_val_acc.extend(history_fold.history['val_accuracy'])

                            # --- جمع‌آوری OOF: پیش‌بینی روی خودِ بلوک val_idx (نه کل آینده) ---
                            # این پیش‌بینی توسط مدلی انجام می‌شود که این بلوک را در آموزش ندیده،
                            # پس برای متا-مدل leak-safe است.
                            fold_val_preds = daughter_model.predict(X_cat_3d[val_idx], batch_size=256, verbose=0).flatten()
                            oof_daughter_preds[val_idx] = fold_val_preds

                            final_history_cat = history_fold
                            fold_idx += 1

                        daughter_model.save(os.path.join(CAT_MODELS_PATH, f"model_{cat_id}.keras"))
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                        
                        ax1.plot(all_val_loss, label='Val Loss', color='orange')
                        ax1.set_title(f'Daughter {cat_id} - Final Chain Loss')
                        ax1.legend()

                        
                        ax2.plot(all_val_acc, label='Val Acc', color='red')

                        ax2.set_title(f'Daughter {cat_id} - Final Chain Accuracy')
                        ax2.legend()

                        plt.savefig(os.path.join(MODELS_DIR, f"daughter_{cat_id}_performance_chart.jpg"))
                        plt.close()
                        
                        print(Fore.GREEN + f"✅ Recursive Training for Category {cat_id} COMPLETED and Saved.")

                        # ذخیره OOF مدل دختر این دسته (هم برای دیباگ، هم برای بازآموزی بعدی متا)
                        joblib.dump(
                            {'positions': group_abs_positions, 'oof': oof_daughter_preds},
                            os.path.join(CAT_MODELS_PATH, f"oof_daughter_{cat_id}.pkl")
                        )

                        # =========================================================================
                        # 🆕 [FIX #2] کالیبراسیون زودهنگام مدل دختر (بلافاصله بعد از OOF، نه بعداً
                        # در Phase 3 روی df_last_month).
                        # باگ قبلی: متا-مدل این‌جا (Phase 2) روی خروجی خامِ oof_daughter_preds
                        # آموزش می‌دید، ولی در تمام نقاط inference (Phase 3/4/5/Live) همیشه
                        # نسخه‌ی کالیبره‌شده‌ی خروجی مدل دختر به آن داده می‌شد -> Train/Serve
                        # Skew واقعی و فعال (نه فرضی). با ساختن کالیبراتور همین‌جا از روی OOF
                        # (leak-safe، چون هر بلوک توسط مدلی پیش‌بینی شده که آن را ندیده) و
                        # ذخیره‌ی آن با همان نامی که Phase 3/4/5/Live از دیسک لود می‌کنند
                        # (calibrator_{cat_id}.pkl)، از این به بعد یک کالیبراتور واحد در همه‌جا
                        # استفاده می‌شود و متا-مدل دقیقاً روی همان مقیاسی آموزش می‌بیند که در
                        # inference می‌بیند.
                        # نکته‌ی صادقانه: این کالیبراتور روی OOF fold-به-fold ساخته می‌شود، نه
                        # روی یک تکه‌ی کاملاً دست‌نخورده مثل df_last_month؛ چون مدل نهایی که در
                        # پایان تمام فولدها را دیده کمی «بیشتر آموزش‌دیده» از مدل‌های میانی هر
                        # فولد است، این کالیبراسیون یک تقریب بسیار خوب است نه یک انطباق ۱۰۰٪
                        # هندسی - اما دقیقاً همین رویکرد استاندارد stacking/OOF-calibration است
                        # و به‌مراتب درست‌تر از ناهماهنگی خام/کالیبره‌ی قبلی است.
                        # =========================================================================
                        daughter_calibrator = calibrate_output(oof_daughter_preds, y_cat)
                        oof_daughter_calibrated = oof_daughter_preds.copy()
                        if daughter_calibrator is not None:
                            joblib.dump(daughter_calibrator, os.path.join(MODELS_DIR, f"calibrator_{cat_id}.pkl"))
                            valid_oof_mask = ~np.isnan(oof_daughter_preds)
                            if valid_oof_mask.any():
                                oof_daughter_calibrated[valid_oof_mask] = daughter_calibrator.predict_proba(
                                    oof_daughter_preds[valid_oof_mask].reshape(-1, 1)
                                )[:, 1]
                            print(Fore.GREEN + f"✅ Calibrator دسته {cat_id} زودهنگام از روی OOF ساخته شد ({int(valid_oof_mask.sum())} ردیف معتبر) و ذخیره شد.")
                        else:
                            print(Fore.YELLOW + f"⚠️ کالیبراسیون زودهنگام دسته {cat_id} ممکن نشد (فقط یک کلاس در OOF). Phase 3 در صورت نیاز یک کالیبراتور جایگزین می‌سازد.")

                        # =========================================================================
                        # --- آموزش متا-مدل این دسته: ورودی = [OOF دختر (کالیبره‌شده), OOF روند] -> هدف = برد/باخت واقعی ---
                        # اگر مدل روند فعال نباشد یا داده‌ی معتبر کافی نباشد، متا-مدل ساخته نمی‌شود
                        # و کل سیستم (Phase 3/4/5) به‌طور خودکار فقط از خروجی خام مدل دختر استفاده می‌کند.
                        # =========================================================================
                        if oof_trend_full is not None:
                            oof_trend_this_cat = oof_trend_full[group_abs_positions]
                            meta_X = build_meta_features(oof_daughter_calibrated, oof_trend_this_cat)
                            meta_y = np.asarray(y_cat)

                            valid_meta_mask = ~np.isnan(meta_X).any(axis=1)
                            n_valid_meta = int(valid_meta_mask.sum())

                            if n_valid_meta >= META_MODEL_MIN_ROWS and len(np.unique(meta_y[valid_meta_mask])) >= 2:
                                meta_model = LogisticRegression(max_iter=1000)
                                meta_model.fit(meta_X[valid_meta_mask], meta_y[valid_meta_mask])
                                joblib.dump(meta_model, os.path.join(META_MODELS_PATH, f"meta_model_{cat_id}.pkl"))
                                print(Fore.GREEN + f"✅ Meta-Model (Daughter + Trend) for Cat {cat_id} trained on {n_valid_meta} valid OOF rows.")

                                # =========================================================================
                                # 🆕 [VIZ #7] نمودار ضرایب متا-مدل. چون متا-مدل یک LogisticRegression روی
                                # [daughter_prob, trend_prob, agreement, disagreement] است، ضریب هر فیچر
                                # مستقیماً نشان می‌دهد متا-مدل چقدر به مدل دختر در مقابل مدل روند اعتماد
                                # می‌کند. اگر ضریب trend_prob تقریباً صفر باشد، یعنی مدل روند عملاً فقط
                                # نویز اضافه می‌کند و ارزش افزوده‌ی واقعی ندارد.
                                # =========================================================================
                                try:
                                    meta_feature_names = ['daughter_prob', 'trend_prob', 'agreement', 'disagreement']
                                    meta_coefs = meta_model.coef_[0]
                                    coef_colors = ['#2ed573' if c >= 0 else '#ff4757' for c in meta_coefs]

                                    plt.figure(figsize=(7, 4.5))
                                    plt.bar(meta_feature_names, meta_coefs, color=coef_colors)
                                    plt.axhline(0, color='black', linewidth=0.8)
                                    plt.title(f'Meta-Model Coefficients - Cat {cat_id}\n(How much weight goes to Daughter vs Trend?)')
                                    plt.ylabel('Logistic Regression Coefficient')
                                    plt.xticks(rotation=15)
                                    plt.tight_layout()
                                    meta_coef_path = os.path.join(META_MODELS_PATH, f"Meta_Model_Coefficients_Cat_{cat_id}.jpg")
                                    plt.savefig(meta_coef_path, dpi=150)
                                    if run_optuna == 0:
                                        plt.show()
                                    else:
                                        plt.close('all')
                                    print(Fore.GREEN + f"✅ نمودار ضرایب متا-مدل دسته {cat_id} ذخیره شد: {meta_coef_path}")
                                except Exception as e:
                                    print(Fore.RED + f"⚠️ خطا در رسم نمودار ضرایب متا-مدل دسته {cat_id}: {e}. رد شد، بدون کرش.")
                            else:
                                print(Fore.YELLOW + f"⚠️ داده‌ی معتبر کافی برای متا-مدل دسته {cat_id} نیست ({n_valid_meta} ردیف). این دسته فقط با خروجی خام مدل دختر کار می‌کند.")
                        else:
                            print(Fore.YELLOW + f"⚠️ مدل روند فعال نیست؛ متا-مدل برای دسته {cat_id} ساخته نشد.")

                else:
                    print(Fore.WHITE + "--- Step 2: Daughter Models loaded from disk (Skipping training) ---")


















    # --- PHASE 3: Threshold Optimization (Advanced Gemini Alpha Score) -------------------------------------------------------------- PHASE 3


    if th_choice in [0, 1, 2]:
        print(Fore.MAGENTA + "\n=== PHASE 3: Threshold Optimization (Detailed Analysis) ===")

        if os.path.exists(DATA_SET_MAH_AKHAR_PATH):
            try:
                best_thresholds = joblib.load(os.path.join(MODELS_DIR, "best_thresholds.pkl"))
            except:
                best_thresholds = {}

            df_last_month = pd.read_csv(DATA_SET_MAH_AKHAR_PATH, sep=';')
            df_last_month.columns = [str(c).strip() for c in df_last_month.columns]

            total_rows_before_cut = len(df_last_month)
            CALIB_SIZE = max(2000, int(total_rows_before_cut * 0.10))
            df_last_month = df_last_month.iloc[-CALIB_SIZE:].copy()

            total_rows_file = len(df_last_month)

            print(Fore.WHITE + f"📊 Total Data Rows: {total_rows_file}")

            all_predictions = np.zeros(total_rows_file)
            y_real_all = df_last_month.iloc[:, 3].values
            cat_ids_all = df_last_month.iloc[:, 0].astype(int).values

            stats = {
                '0': {'bal': 1000.0, 'peak': 1000.0, 'mdd': 0, 'wins': 0, 'loss': 0, 'trades': 0, 'total_rows': 0, 'th': 0, 'kpi': 0},
                '1': {'bal': 1000.0, 'peak': 1000.0, 'mdd': 0, 'wins': 0, 'loss': 0, 'trades': 0, 'total_rows': 0, 'th': 0, 'kpi': 0},
                'combined': {'bal': 1000.0, 'peak': 1000.0, 'mdd': 0, 'wins': 0, 'loss': 0, 'trades': 0}
            }

            # مرحله ۱: پیش‌بینی دسته‌ای (Batch Prediction) برای کل فایل (بسیار سریع)
            for cat_id_str in ['0', '1']:
                mask = (cat_ids_all == int(cat_id_str))
                group_indices = np.where(mask)[0]
                stats[cat_id_str]['total_rows'] = len(group_indices)

                if len(group_indices) == 0:
                    print(Fore.YELLOW + f"ℹ️ هیچ ردیفی برای دسته {cat_id_str} در این فایل وجود ندارد. (مثلاً هنوز معامله‌ی SELL نداشتید). این دسته رد می‌شود.")
                    continue

                # 🛡️ محافظت در برابر نبود مدل/اسکیلر این دسته (مثلاً مدل SELL هنوز آموزش داده نشده)
                model_path_cat = os.path.join(CAT_MODELS_PATH, f"model_{cat_id_str}.keras")
                scaler_path_cat = os.path.join(SCALERS_PATH, f"scaler_{cat_id_str}.pkl")
                if not (os.path.exists(model_path_cat) and os.path.exists(scaler_path_cat)):
                    print(Fore.YELLOW + f"⚠️ مدل/اسکیلر دسته {cat_id_str} پیدا نشد (هنوز آموزش داده نشده). این دسته در Phase 3 رد می‌شود، بدون کرش.")
                    continue

                try:
                    # ۱. لود لیست فیچرهای واحد (مشترک برای مادر و دخترها)
                    union_feat_path = os.path.join(MODELS_DIR, "union_features_mother.pkl")
                    current_feats = joblib.load(union_feat_path)
                    
                    # ۲. محاسبه تعداد خانواده‌ها برای Reshape دقیق
                    current_tk = len(current_feats) // TIME_STEPS
                    
                    # ۳. جداسازی دیتای مربوط به این دسته با فیچرهای درست
                    X_cat_raw = df_last_month.iloc[group_indices][current_feats]

                    scaler_cat = joblib.load(scaler_path_cat)
                    X_cat_scaled = scaler_cat.transform(X_cat_raw.values)
                    
                    X_cat_3d = X_cat_scaled.reshape((-1, current_tk, TIME_STEPS))
                    X_cat_3d = np.transpose(X_cat_3d, (0, 2, 1))

                    model_cat = tf.keras.models.load_model(model_path_cat,
                                      custom_objects={'MultiHeadSelfAttention': MultiHeadSelfAttention})
                    preds = model_cat.predict(X_cat_3d, batch_size=256, verbose=0).flatten()

                    # --- بخش کالیبراسیون ---
                    # 🆕 [FIX #2] اول تلاش می‌کنیم همان کالیبراتوری که در Phase 2 (روی OOF،
                    # leak-safe) ساخته و ذخیره شده را لود کنیم - دقیقاً همان کالیبراتوری که
                    # متا-مدل هم رویش آموزش دیده است. این یعنی دیگر یک کالیبراتور جدید و
                    # متفاوت هر بار روی df_last_month ساخته نمی‌شود، و ورودی متا-مدل در
                    # train و inference کاملاً هم‌مقیاس می‌ماند.
                    # اگر (مثلاً به‌خاطر اجرای قدیمی/داده ناکافی در Phase 2) این فایل پیدا
                    # نشد، دقیقاً مثل قبل، یک کالیبراتور جایگزین روی df_last_month ساخته
                    # می‌شود تا برنامه کرش نکند (در این حالت fallback ممکن است یک ناهم‌مقیاسی
                    # جزئی نسبت به متا-مدل باقی بماند - توصیه می‌شود دخترها دوباره آموزش ببینند).
                    cal_path_cat = os.path.join(MODELS_DIR, f"calibrator_{cat_id_str}.pkl")
                    print(Fore.YELLOW + f"⚖️ Calibrating Model Outputs for Cat {cat_id_str}...")
                    if os.path.exists(cal_path_cat):
                        calibrator = joblib.load(cal_path_cat)
                        print(Fore.CYAN + f"   ⚡ کالیبراتور دسته {cat_id_str} از Phase 2 (OOF) لود شد (همان کالیبراتوری که متا-مدل رویش آموزش دیده).")
                    else:
                        calibrator = calibrate_output(preds, y_real_all[group_indices])
                        if calibrator is not None:
                            joblib.dump(calibrator, cal_path_cat)
                        print(Fore.YELLOW + f"   ⚠️ کالیبراتور Phase 2 برای دسته {cat_id_str} پیدا نشد؛ یک کالیبراتور جایگزین روی df_last_month ساخته شد.")
                    if calibrator is not None:
                        preds = calibrator.predict_proba(preds.reshape(-1, 1))[:, 1]

                    # --- 🆕 ترکیب با مدل روند از طریق متا-مدل (اگر موجود باشد) ---
                    trend_probs_cat = predict_trend_probability(df_last_month.iloc[group_indices])
                    preds, used_meta = combine_daughter_and_trend(preds, trend_probs_cat, cat_id_str)
                    if used_meta:
                        print(Fore.GREEN + f"   ✅ خروجی نهایی دسته {cat_id_str} از متا-مدل (دختر + روند) گرفته شد.")
                    else:
                        print(Fore.WHITE + f"   ℹ️ متا-مدل برای دسته {cat_id_str} در دسترس نیست؛ فقط خروجی خام مدل دختر استفاده می‌شود.")

                    # آپدیت کردن آرایه اصلی برای استفاده در شبیه‌سازی ترکیبی نهایی
                    all_predictions[group_indices] = preds

                    # بهینه‌سازی تریشلد برای این دسته
                    if th_choice == 2 or (th_choice == 0 and int(cat_id_str) == 0) or (th_choice == 1 and int(cat_id_str) == 1):
                        print(Fore.YELLOW + f"🔎 Density-Based Optimization for Cat {cat_id_str}...")
                        
                        y_sub = y_real_all[group_indices]
                        all_win_trade = np.sum(y_sub > 0)
                        all_loss_trade = np.sum(y_sub <= 0)
                        total_potential = len(y_sub)

                        # =========================================================================
                        # 🆕 [VIZ #3] هیستوگرام توزیع احتمالات پیش‌بینی برای Win در مقابل Loss.
                        # مستقل از انتخاب threshold، این نشان می‌دهد مدل چقدر ذاتاً این دو کلاس را
                        # از هم جدا می‌کند. اگر دو توزیع خیلی روی هم بیفتند، یعنی جداسازی مدل
                        # ضعیف است و کل عملکرد دارد روی threshold-tuning سوار می‌شود (شکننده).
                        # =========================================================================
                        try:
                            plt.figure(figsize=(9, 5))
                            plt.hist(preds[y_sub > 0], bins=30, alpha=0.55, label=f'Win (n={int(all_win_trade)})', color='#2ed573')
                            plt.hist(preds[y_sub <= 0], bins=30, alpha=0.55, label=f'Loss (n={int(all_loss_trade)})', color='#ff4757')
                            plt.xlabel('Predicted Probability')
                            plt.ylabel('Count')
                            plt.title(f'Prediction Distribution: Win vs Loss - Cat {cat_id_str}')
                            plt.legend()
                            plt.tight_layout()
                            hist_path = os.path.join(MODELS_DIR, f"Prediction_Distribution_Cat_{cat_id_str}.jpg")
                            plt.savefig(hist_path, dpi=150)
                            if run_optuna == 0:
                                plt.show()
                            else:
                                plt.close('all')
                            print(Fore.GREEN + f"✅ هیستوگرام توزیع Win/Loss دسته {cat_id_str} ذخیره شد: {hist_path}")
                        except Exception as e:
                            print(Fore.RED + f"⚠️ خطا در رسم هیستوگرام توزیع دسته {cat_id_str}: {e}. رد شد، بدون کرش.")

                        # =========================================================================
                        # 🆕 [VIZ #4] ROC Curve و Precision-Recall Curve. این‌ها مستقل از هر
                        # threshold خاصی، قدرت ذاتی تفکیک مدل (discriminative power) را با یک
                        # عدد AUC خلاصه می‌کنند - معیار سریع و استاندارد برای مقایسه‌ی Trial های
                        # مختلف Optuna با هم.
                        # =========================================================================
                        try:
                            y_bin_sub = (y_sub > 0).astype(int)
                            if len(np.unique(y_bin_sub)) == 2:
                                fpr, tpr, _ = roc_curve(y_bin_sub, preds)
                                roc_auc = auc(fpr, tpr)
                                prec, rec, _ = precision_recall_curve(y_bin_sub, preds)
                                pr_auc = auc(rec, prec)

                                fig, (axr, axp) = plt.subplots(1, 2, figsize=(12, 5))
                                axr.plot(fpr, tpr, color='#1e90ff', linewidth=2, label=f'AUC = {roc_auc:.3f}')
                                axr.plot([0, 1], [0, 1], linestyle='--', color='gray')
                                axr.set_xlabel('False Positive Rate')
                                axr.set_ylabel('True Positive Rate')
                                axr.set_title(f'ROC Curve - Cat {cat_id_str}')
                                axr.legend()

                                axp.plot(rec, prec, color='#ff6348', linewidth=2, label=f'AUC = {pr_auc:.3f}')
                                axp.axhline(all_win_trade / total_potential if total_potential > 0 else 0, linestyle='--', color='gray', label='Baseline (Win Rate)')
                                axp.set_xlabel('Recall')
                                axp.set_ylabel('Precision')
                                axp.set_title(f'Precision-Recall Curve - Cat {cat_id_str}')
                                axp.legend()

                                plt.tight_layout()
                                roc_path = os.path.join(MODELS_DIR, f"ROC_PR_Curve_Cat_{cat_id_str}.jpg")
                                plt.savefig(roc_path, dpi=150)
                                if run_optuna == 0:
                                    plt.show()
                                else:
                                    plt.close('all')
                                print(Fore.GREEN + f"✅ ROC/PR Curve دسته {cat_id_str} ذخیره شد (ROC-AUC={roc_auc:.3f} | PR-AUC={pr_auc:.3f}): {roc_path}")
                            else:
                                print(Fore.YELLOW + f"ℹ️ فقط یک کلاس در داده‌ی دسته {cat_id_str} وجود دارد؛ رسم ROC/PR معنا ندارد.")
                        except Exception as e:
                            print(Fore.RED + f"⚠️ خطا در رسم ROC/PR Curve دسته {cat_id_str}: {e}. رد شد، بدون کرش.")

                        low_v = np.percentile(preds, 10)
                        high_v = np.percentile(preds, 95)
                        search_space = np.linspace(low_v, high_v, 150) 

                        best_th = 0.5
                        best_kpi = -99999

                        # 🆕 [VIZ #5] ذخیره‌ی کل منحنی KPI برای هر threshold امتحان‌شده (نه فقط بهترین)
                        kpi_curve_th_list = []
                        kpi_curve_val_list = []

                        print(f"DEBUG {cat_id_str}: Density Range: {low_v:.3f} to {high_v:.3f}")

                        for th in search_space:
                            tmp_bal, tmp_peak, tmp_mdd = 1000.0, 1000.0, 0.0
                            win_f, loss_f = 0, 0
                            
                            for p, a in zip(preds, y_sub):
                                if p >= th:
                                    if a > 0:
                                        tmp_bal += tmp_bal * 0.01
                                        win_f += 1
                                    else:
                                        tmp_bal -= tmp_bal * 0.01
                                        loss_f += 1
                                    
                                    tmp_peak = max(tmp_peak, tmp_bal)
                                    drawdown = (tmp_peak - tmp_bal) / tmp_peak if tmp_peak > 0 else 0
                                    tmp_mdd = max(tmp_mdd, drawdown)

                            if (win_f + loss_f) < (total_potential * MIN_TRADE_PERCENT):
                                continue

                            PP = (tmp_bal - 1000) / 1000
                            MDD_val = tmp_mdd if tmp_mdd > 0.03 else 0.03
                            
                            ratio_win = (win_f / all_win_trade) if all_win_trade > 0 else 0
                            ratio_loss_saved = (all_loss_trade - loss_f) / all_loss_trade if all_loss_trade > 0 else 0
                            
                            current_kpi = (PP / MDD_val) * (ratio_win**2) * ratio_loss_saved 

                            kpi_curve_th_list.append(th)
                            kpi_curve_val_list.append(current_kpi)

                            if current_kpi > best_kpi:
                                best_kpi, best_th = current_kpi, th
                                stats[cat_id_str].update({
                                    'th': best_th, 'kpi': best_kpi, 'bal': tmp_bal, 
                                    'mdd': tmp_mdd, 'wins': win_f, 'loss': loss_f, 'trades': (win_f + loss_f)
                                })
                                print(Fore.WHITE + f"  ⭐ New Best TH: {th:.3f} | KPI: {current_kpi:.3f} | WinRate: {(win_f/(win_f+loss_f)*100):.1f}% | Trades: {win_f+loss_f}")
                    
                        best_thresholds[cat_id_str] = round(float(best_th), 3)

                        # =========================================================================
                        # 🆕 [VIZ #5] رسم منحنی KPI بر حسب Threshold. یک پیک تیز و باریک یعنی
                        # threshold انتخابی شکننده است (با کوچک‌ترین تغییر رژیم بازار می‌افتد)؛
                        # یک پلاتوی پهن یعنی انتخاب robust و قابل‌اعتمادتر برای Live.
                        # =========================================================================
                        try:
                            if len(kpi_curve_th_list) > 0:
                                plt.figure(figsize=(10, 5))
                                plt.plot(kpi_curve_th_list, kpi_curve_val_list, color='#1e90ff', linewidth=1.8)
                                plt.axvline(best_th, color='red', linestyle='--', label=f'Best TH = {best_th:.3f}')
                                plt.xlabel('Threshold')
                                plt.ylabel('KPI')
                                plt.title(f'KPI Sensitivity to Threshold - Cat {cat_id_str}\n(Sharp peak = fragile | Wide plateau = robust)')
                                plt.legend()
                                plt.grid(True, alpha=0.25)
                                plt.tight_layout()
                                th_sens_path = os.path.join(MODELS_DIR, f"Threshold_Sensitivity_Cat_{cat_id_str}.jpg")
                                plt.savefig(th_sens_path, dpi=150)
                                if run_optuna == 0:
                                    plt.show()
                                else:
                                    plt.close('all')
                                print(Fore.GREEN + f"✅ نمودار حساسیت KPI به Threshold دسته {cat_id_str} ذخیره شد: {th_sens_path}")
                            else:
                                print(Fore.YELLOW + "ℹ️ هیچ Threshold ای شرط حداقل حجم معامله را رد نکرد؛ منحنی حساسیت رسم نشد.")
                        except Exception as e:
                            print(Fore.RED + f"⚠️ خطا در رسم منحنی حساسیت Threshold دسته {cat_id_str}: {e}. رد شد، بدون کرش.")

                except Exception as e:
                    print(Fore.RED + f"❌ خطا در پردازش دسته {cat_id_str} در Phase 3: {e}. این دسته رد می‌شود، بقیه‌ی سیستم ادامه می‌دهد.")
                    continue

            # مرحله ۲: شبیه‌سازی ترکیبی نهایی (حالا با سرعت نور چون مدل صدا زده نمی‌شود)
            print(Fore.GREEN + "🔄 Running Final Combined Simulation (Vectorized)...")
            for i in range(total_rows_file):
                c_id = str(cat_ids_all[i])
                if c_id not in ['0', '1']: continue
                
                pred = all_predictions[i]
                y_val = y_real_all[i]
                th = best_thresholds.get(c_id, 0.5)

                if pred >= th:
                    stats['combined']['trades'] += 1
                    if y_val > 0:
                        stats['combined']['wins'] += 1
                        stats['combined']['bal'] += stats['combined']['bal'] * 0.01
                    else:
                        stats['combined']['bal'] -= stats['combined']['bal'] * 0.01

                    stats['combined']['peak'] = max(stats['combined']['peak'], stats['combined']['bal'])
                    stats['combined']['mdd'] = max(stats['combined']['mdd'], (stats['combined']['peak']-stats['combined']['bal'])/stats['combined']['peak'])

            joblib.dump(best_thresholds.get('0', 0.5), os.path.join(MODELS_DIR, "th_sell_0.pkl"))
            joblib.dump(best_thresholds.get('1', 0.5), os.path.join(MODELS_DIR, "th_buy_1.pkl"))
            print(Fore.GREEN + "✅ تریشلدها به صورت فایل‌های مجزا (th_sell_0 و th_buy_1) ذخیره شدند.")

            def get_report_line(name, d):
                wr = (d['wins']/d['trades']*100) if d['trades']>0 else 0
                prof = (d['bal']-1000)/1000*100
                return f"{name:<15} | {d['trades']:<7} | {wr:<6.2f}% | {prof:<8.2f}% | {d['mdd']*100:<7.2f}% | {d['bal']:<8.2f}$"

            report = f"""
    ==========================================================================
                    GEMINI DETAILED PHASE 3 REPORT
    ==========================================================================
    MODEL           | TRADES  | WINRATE | PROFIT % | MAX DD % | FINAL BAL
    --------------------------------------------------------------------------
    {get_report_line('COMBINED (ALL)', stats['combined'])}
    {get_report_line('SELL (Cat 0)', stats['0'])}
    {get_report_line('BUY  (Cat 1)', stats['1'])}
    --------------------------------------------------------------------------
    DETAILS:
    > Total File Rows    : {total_rows_file}
    > Sell Rows in File  : {stats['0']['total_rows']}
    > Buy Rows in File   : {stats['1']['total_rows']}
    > Chosen Thresholds  : SELL={best_thresholds.get('0',0.5)} | BUY={best_thresholds.get('1',0.5)}
    ==========================================================================
    """
            print(Fore.CYAN + report)
            with open(os.path.join(MODELS_DIR, "Phase3_Detailed_Report.txt"), "w", encoding="utf-8") as f:
                f.write(report)
















    # --- PHASE 4: Professional Comparative Analysis (Gemini Alpha Logic) ---       -------------------------------------------------------     PHASE 4


    if run_test == 1:
        print(Fore.MAGENTA + "\n" + "="*50)
        print(Fore.MAGENTA + "=== PHASE 4: VISUAL & STATISTICAL ANALYSIS (Alpha) ===")
        print(Fore.MAGENTA + "="*50)

        if os.path.exists(TEST_DATA_PATH):
            df_test_full = pd.read_csv(TEST_DATA_PATH, sep=';')
            df_test_full.columns = [str(c).strip() for c in df_test_full.columns]

            # 🆕 تقسیم «یک روز در میون» (نه ردیف در میون) به دو گروه Validation
            # و Final-Test، بر اساس ستون TradeTimestamp (فرمت: 2024.05.24 19:05).
            # همه‌ی ردیف‌های یک روز معاملاتی با هم توی یک گروه می‌مونن؛ روزها
            # به‌ترتیب تاریخ، یکی‌درمیون بین دو گروه پخش می‌شن. چون پنجره‌ی
            # لوک‌بک ۳۰-تایی فیچرها در مقیاس ساعتی‌ـه (نه چند-روزه)، این کار
            # عملاً هم‌پوشانی بین Validation و Final-Test رو از بین می‌بره.
            try:
                parsed_dates = pd.to_datetime(df_test_full['TradeTimestamp'], format='%Y.%m.%d %H:%M').dt.date
                unique_days = sorted(parsed_dates.unique())
                day_to_group = {day: (i % 2) for i, day in enumerate(unique_days)}
                row_group = parsed_dates.map(day_to_group)

                df_validation    = df_test_full[row_group == 0].reset_index(drop=True)
                df_final_holdout = df_test_full[row_group == 1].reset_index(drop=True)

                n_val_days = sum(1 for g in day_to_group.values() if g == 0)
                n_test_days = sum(1 for g in day_to_group.values() if g == 1)
                print(Fore.CYAN + f"📅 تقسیم روزانه انجام شد: {len(unique_days)} روز معاملاتی → {n_val_days} روز Validation / {n_test_days} روز Final-Test")
            except Exception as e:
                # 🛡️ اگه ستون TradeTimestamp پیدا/پارس نشد (مثلاً فرمت فرق داشت)،
                # به‌عنوان جایگزین امن به تفکیک ردیف‌به‌ردیف برمی‌گردیم تا کل
                # پایپ‌لاین کرش نکنه
                print(Fore.RED + f"⚠️ خطا در پارس کردن ستون TradeTimestamp ({e}) - برگشت موقت به تفکیک ردیف‌به‌ردیف.")
                df_validation    = df_test_full.iloc[0::2].reset_index(drop=True)
                df_final_holdout = df_test_full.iloc[1::2].reset_index(drop=True)

            if USE_FINAL_HOLDOUT_TEST:
                df_test = df_final_holdout
                print(Fore.MAGENTA + f"🔒 ارزیابی نهایی روی Final-Test کاملاً دست‌نخورده ({len(df_test)} ردیف) - صادقانه‌ترین تخمین عملکرد واقعی.")
            else:
                df_test = df_validation
                print(Fore.CYAN + f"🧪 ارزیابی این Trial روی Validation ({len(df_test)} ردیف).")

            y_real = df_test.iloc[:, 3].values
            cat_ids = df_test.iloc[:, 0].astype(int).values
            total_rows = len(df_test)

            report_data = {
                'combined': {'base_bal': 1000.0, 'ai_bal': 1000.0, 'base_wins': 0, 'ai_wins': 0, 'base_tr': 0, 'ai_tr': 0, 'base_peak': 1000.0, 'ai_peak': 1000.0, 'base_mdd': 0, 'ai_mdd': 0, 'saved_loss': 0, 'pot_loss': 0},
                '0':        {'base_bal': 1000.0, 'ai_bal': 1000.0, 'base_wins': 0, 'ai_wins': 0, 'base_tr': 0, 'ai_tr': 0, 'base_peak': 1000.0, 'ai_peak': 1000.0, 'base_mdd': 0, 'ai_mdd': 0, 'saved_loss': 0, 'pot_loss': 0},
                '1':        {'base_bal': 1000.0, 'ai_bal': 1000.0, 'base_wins': 0, 'ai_wins': 0, 'base_tr': 0, 'ai_tr': 0, 'base_peak': 1000.0, 'ai_peak': 1000.0, 'base_mdd': 0, 'ai_mdd': 0, 'saved_loss': 0, 'pot_loss': 0}
            }

            # ۱. انجام پیش‌بینی دسته‌ای برای سرعت (Batch Prediction)
            all_test_preds = np.zeros(total_rows)
            available_categories = []   # 🆕 فقط دسته‌هایی که واقعاً مدل/داده دارند اینجا اضافه می‌شوند

            best_thresholds = {}
            try:
                best_thresholds['0'] = joblib.load(os.path.join(MODELS_DIR, "th_sell_0.pkl"))
            except:
                pass
            try:
                best_thresholds['1'] = joblib.load(os.path.join(MODELS_DIR, "th_buy_1.pkl"))
            except:
                pass
            if '0' not in best_thresholds: best_thresholds['0'] = 0.5
            if '1' not in best_thresholds: best_thresholds['1'] = 0.5

            for c_id in ['0', '1']:
                mask = (cat_ids == int(c_id))
                indices = np.where(mask)[0]

                if len(indices) == 0:
                    print(Fore.YELLOW + f"ℹ️ هیچ ردیفی برای دسته {c_id} در فایل تست وجود ندارد. (مثلاً هنوز معامله‌ی SELL نداشتید). این دسته در Phase 4 رد می‌شود.")
                    continue

                # 🛡️ محافظت در برابر نبود مدل/اسکیلر این دسته - به‌جای exit() فقط این دسته را رد می‌کنیم
                m_path = os.path.join(CAT_MODELS_PATH, f"model_{c_id}.keras")
                s_path = os.path.join(SCALERS_PATH, f"scaler_{c_id}.pkl")
                if not (os.path.exists(m_path) and os.path.exists(s_path)):
                    print(Fore.YELLOW + f"⚠️ مدل/اسکیلر دسته {c_id} پیدا نشد (هنوز آموزش داده نشده). این دسته در Phase 4 رد می‌شود، بدون توقف کل برنامه.")
                    continue

                try:
                    union_feat_path = os.path.join(MODELS_DIR, "union_features_mother.pkl")
                    current_feat_names = joblib.load(union_feat_path)
                    
                    current_model_top_k = len(current_feat_names) // TIME_STEPS
                    
                    scaler = joblib.load(s_path)
                    model = tf.keras.models.load_model(m_path, 
                                                      custom_objects={'MultiHeadSelfAttention': MultiHeadSelfAttention})
                    
                    X_cat_test_raw = df_test.iloc[indices][current_feat_names] 
                    X_test_scaled = scaler.transform(X_cat_test_raw.values)

                    X_3d = X_test_scaled.reshape(-1, current_model_top_k, TIME_STEPS)
                    X_3d = np.transpose(X_3d, (0, 2, 1))
                    
                    raw_test_preds = model.predict(X_3d, batch_size=256, verbose=0).flatten()

                    cal_path = os.path.join(MODELS_DIR, f"calibrator_{c_id}.pkl")
                    if os.path.exists(cal_path):
                        calibrator = joblib.load(cal_path)
                        cat_preds = calibrator.predict_proba(raw_test_preds.reshape(-1, 1))[:, 1]
                    else:
                        cat_preds = raw_test_preds

                    # --- 🆕 ترکیب با مدل روند از طریق متا-مدل (اگر موجود باشد) ---
                    trend_probs_cat = predict_trend_probability(df_test.iloc[indices])
                    cat_preds, used_meta = combine_daughter_and_trend(cat_preds, trend_probs_cat, c_id)
                    if used_meta:
                        print(Fore.GREEN + f"   ✅ خروجی نهایی دسته {c_id} در Phase 4 از متا-مدل (دختر + روند) گرفته شد.")
                    else:
                        print(Fore.WHITE + f"   ℹ️ متا-مدل برای دسته {c_id} در دسترس نیست؛ فقط خروجی خام مدل دختر در Phase 4 استفاده می‌شود.")

                    all_test_preds[indices] = cat_preds
                    available_categories.append(c_id)

                except Exception as e:
                    print(Fore.RED + f"❌ خطا در پردازش دسته {c_id} در Phase 4: {e}. این دسته رد می‌شود، بقیه‌ی سیستم ادامه می‌دهد.")
                    continue

            if len(available_categories) == 0:
                print(Fore.RED + "❌ هیچ مدلی برای هیچ دسته‌ای در دسترس نیست. Phase 4 متوقف می‌شود (بدون کرش کل برنامه).")
            else:
                # ۲. شبیه‌سازی گام‌به‌گام (Step-by-Step Simulation)
                print(Fore.CYAN + "📊 Calculating metrics for Combined, Sell and Buy...")

                base_history = [1000.0]
                ai_history = [1000.0]
                
                all_win_test = np.sum(y_real > 0)
                all_loss_test = np.sum(y_real <= 0)

                for i in range(total_rows):
                    c_str = str(cat_ids[i])
                    if c_str not in available_categories: continue
                    
                    y_val = y_real[i]
                    is_win = y_val > 0
                    pred = all_test_preds[i]
                    th = best_thresholds.get(c_str, 0.5)

                    target_keys = [c_str, 'combined']

                    for k in target_keys:
                        report_data[k]['base_tr'] += 1
                        if is_win:
                            report_data[k]['base_wins'] += 1
                            report_data[k]['base_bal'] += report_data[k]['base_bal'] * 0.01
                        else:
                            report_data[k]['base_bal'] -= report_data[k]['base_bal'] * 0.01
                        
                        if not is_win: report_data[k]['pot_loss'] += 1

                        if pred >= th:
                            report_data[k]['ai_tr'] += 1
                            if is_win:
                                report_data[k]['ai_wins'] += 1
                                report_data[k]['ai_bal'] += report_data[k]['ai_bal'] * 0.01
                            else:
                                report_data[k]['ai_bal'] -= report_data[k]['ai_bal'] * 0.01
                        else:
                            if not is_win: report_data[k]['saved_loss'] += 1

                        report_data[k]['base_peak'] = max(report_data[k]['base_peak'], report_data[k]['base_bal'])
                        report_data[k]['base_mdd'] = max(report_data[k]['base_mdd'], (report_data[k]['base_peak']-report_data[k]['base_bal'])/report_data[k]['base_peak'])
                        report_data[k]['ai_peak'] = max(report_data[k]['ai_peak'], report_data[k]['ai_bal'])
                        report_data[k]['ai_mdd'] = max(report_data[k]['ai_mdd'], (report_data[k]['ai_peak']-report_data[k]['ai_bal'])/report_data[k]['ai_peak'])

                    base_history.append(report_data['combined']['base_bal'])
                    ai_history.append(report_data['combined']['ai_bal'])




                # --- تابع محاسباتی مرجع برای KPI یکپارچه با جریمه ---
                # 🆕 [SHADOW EVAL] calculate_unified_kpi به سطح ماژول منتقل شد (بالای
                # فایل، کنار run_honesty_check) تا evaluate_dataset_kpi هم بتواند از
                # همان فرمول دقیق استفاده کند - بدون این تغییر، اگر فرمول KPI بعداً
                # عوض می‌شد، یکی از این دو محل ممکن بود جا بماند و دو معیار ناهماهنگ
                # تولید کند.
                def format_section(title, d):
                    b_wr = (d['base_wins']/d['base_tr']*100) if d['base_tr']>0 else 0
                    a_wr = (d['ai_wins']/d['ai_tr']*100) if d['ai_tr']>0 else 0
                    la_rate = (d['saved_loss']/d['pot_loss']*100) if d['pot_loss']>0 else 0
                    net_ai = d['ai_bal'] - 1000
                    net_base = d['base_bal'] - 1000
                    
                    # محاسبه KPI یکپارچه
                    ggi = calculate_unified_kpi(d)        

                    return f"""
    [{title}]
    METRIC                | BASELINE (NO AI) | AI FILTERED      | IMPROVEMENT
    --------------------------------------------------------------------------
    Total Trades          | {d['base_tr']:<16} | {d['ai_tr']:<16} | {((d['ai_tr']/d['base_tr']*100) if d['base_tr']>0 else 0):.1f}% Volume
    Win Rate              | {b_wr:<15.2f}% | {a_wr:<15.2f}% | {(a_wr - b_wr):+.2f}%
    Net Profit            | {net_base:<15.2f}$ | {net_ai:<15.2f}$ | {(net_ai - net_base):+.2f}$
    Max Drawdown          | {d['base_mdd']*100:<15.2f}% | {d['ai_mdd']*100:<15.2f}% | {(d['base_mdd']*100 - d['ai_mdd']*100):+.2f}%
    --------------------------------------------------------------------------
    AI SPECIAL METRICS:
    > Loss Avoidance Rate: {la_rate:.2f}% (Saved from {d['saved_loss']} losses)
    > KPI : {ggi:.6f}
    """

                sections_text = format_section("COMBINED SYSTEM (BUY + SELL)", report_data['combined'])
                if '0' in available_categories:
                    sections_text += format_section("SELL MODEL ONLY (CAT 0)", report_data['0'])
                else:
                    sections_text += "\n[SELL MODEL ONLY (CAT 0)]\n⚠️ مدل SELL هنوز آموزش داده نشده / داده‌ای برای آن وجود ندارد - این بخش رد شد.\n"
                if '1' in available_categories:
                    sections_text += format_section("BUY MODEL ONLY (CAT 1)", report_data['1'])
                else:
                    sections_text += "\n[BUY MODEL ONLY (CAT 1)]\n⚠️ مدل BUY هنوز آموزش داده نشده / داده‌ای برای آن وجود ندارد - این بخش رد شد.\n"

                final_report = f"""
    ==========================================================================
            GEMINI ALPHA COMPARATIVE PERFORMANCE REPORT
    ==========================================================================
    {sections_text}
    ==========================================================================
    """
                print(Fore.GREEN + final_report)
                report_file_name = "FINAL_HOLDOUT_UNBIASED_Report.txt" if USE_FINAL_HOLDOUT_TEST else "Alpha_Performance_Report.txt"
                with open(os.path.join(MODELS_DIR, report_file_name), "w", encoding="utf-8") as f:
                    f.write(final_report)

                print(Fore.GREEN + f"✅ Detailed report saved to: {report_file_name}")

                
                # --- بخش رسم نمودار مقایسه‌ای Performance ---
                print(Fore.YELLOW + "📈 در حال ترسیم نمودار مقایسه‌ای...")
                
                d_combined = report_data['combined']
                
                # استفاده مستقیم از تابع یکپارچه
                final_kpi_score = calculate_unified_kpi(d_combined)
                optuna_return_score = final_kpi_score

                # =========================================================================
                # 🆕 [SHADOW EVAL] ارزیابی silent روی Final-Test/Holdout - فقط وقتی این یک
                # Trial واقعی از Optuna است (نه اجرای دستی) و داریم روی Validation ارزیابی
                # می‌کنیم (نه در Final Retrain که خودش مستقیماً روی Holdout حساب می‌کند،
                # چون در اون حالت df_test از قبل خودِ df_final_holdout است و انجام دوباره‌ی
                # همین کار تکراری و بی‌فایده است). نتیجه‌ی این ارزیابی به optuna_return_score
                # اضافه/تفریق نمی‌شود و به Optuna داده نمی‌شود - فقط لاگ می‌شود تا خودت
                # بتوانی روند Val KPI در مقابل Holdout KPI را در طول Trial ها دنبال کنی.
                # =========================================================================
                if trial is not None and not USE_FINAL_HOLDOUT_TEST:
                    try:
                        shadow_holdout_kpi, shadow_holdout_d = evaluate_dataset_kpi(df_final_holdout, best_thresholds)
                        log_shadow_holdout_result(trial, final_kpi_score, d_combined, shadow_holdout_kpi, shadow_holdout_d)
                    except Exception as e:
                        print(Fore.RED + f"⚠️ خطا در ارزیابی Shadow Holdout: {e}. رد شد، بدون کرش و بدون تاثیر روی Trial.")



                plt.figure(figsize=(14, 7))
                
                plt.plot(base_history, label='Baseline (Trade All)', color='#ff4757', alpha=0.6, linestyle='--')
                plt.plot(ai_history, label='AI Filtered', color='#2ed573', linewidth=2.5)
                
                plt.fill_between(range(len(ai_history)), ai_history, base_history, 
                                where=(np.array(ai_history) > np.array(base_history)), 
                                color='#2ed573', alpha=0.1)
                
                plt.title(f'Equity Curve Comparison\nFinal KPI: {final_kpi_score:.6f}', fontsize=14)
                plt.xlabel('Trades Count')
                plt.ylabel('Balance ($)')
                plt.legend()
                plt.grid(True, alpha=0.2)
                
                plot_file_name = "FINAL_HOLDOUT_Equity_Curve.jpg" if USE_FINAL_HOLDOUT_TEST else "Baseline__VS__AI Filtered.jpg"
                plot_path = os.path.join(MODELS_DIR, plot_file_name)
                plt.savefig(plot_path, dpi=300)
                
                if run_optuna == 0:
                    plt.show()
                
                else:
                    plt.close('all')
                
                print(Fore.GREEN + f"✅ نمودار مقایسه‌ای ذخیره شد: {plot_path}")

                # =========================================================================
                # 🆕 [VIZ #6] نمودار Underwater / Drawdown - جدا از Equity Curve.
                # Max Drawdown فقط یک عدد (بدترین لحظه) را گزارش می‌کند؛ این نمودار نشان
                # می‌دهد ریسک دقیقاً کِی اتفاق افتاده و چقدر طول کشیده تا balance به peak
                # قبلی برگردد (recovery time) - چیزی که یک عدد تنها نشانش نمی‌دهد.
                # =========================================================================
                try:
                    def _drawdown_series(history):
                        history_arr = np.array(history, dtype='float64')
                        running_peak = np.maximum.accumulate(history_arr)
                        dd = np.where(running_peak > 0, (running_peak - history_arr) / running_peak, 0.0)
                        return dd * 100.0  # درصد

                    base_dd = _drawdown_series(base_history)
                    ai_dd = _drawdown_series(ai_history)

                    plt.figure(figsize=(14, 5))
                    plt.fill_between(range(len(base_dd)), -base_dd, 0, color='#ff4757', alpha=0.35, label=f'Baseline (Max DD: {base_dd.max():.2f}%)')
                    plt.fill_between(range(len(ai_dd)), -ai_dd, 0, color='#2ed573', alpha=0.45, label=f'AI Filtered (Max DD: {ai_dd.max():.2f}%)')
                    plt.title('Underwater Plot (Drawdown Over Time)', fontsize=14)
                    plt.xlabel('Trades Count')
                    plt.ylabel('Drawdown (%)')
                    plt.legend()
                    plt.grid(True, alpha=0.2)

                    dd_file_name = "FINAL_HOLDOUT_Drawdown.jpg" if USE_FINAL_HOLDOUT_TEST else "Baseline__VS__AI_Drawdown.jpg"
                    dd_path = os.path.join(MODELS_DIR, dd_file_name)
                    plt.savefig(dd_path, dpi=200)
                    if run_optuna == 0:
                        plt.show()
                    else:
                        plt.close('all')
                    print(Fore.GREEN + f"✅ نمودار Drawdown (Underwater) ذخیره شد: {dd_path}")
                except Exception as e:
                    print(Fore.RED + f"⚠️ خطا در رسم نمودار Drawdown: {e}. رد شد، بدون کرش.")

                _record_trial_result(trial, optuna_return_score, d_combined)

                # --- برگرداندن امتیاز نهایی (برای Optuna، یا برای نمایش در اجرای دستی) ---
                return optuna_return_score










    # --- اجرای تست در انتهای فاز ۴ ---
    # 🛡️ فقط برای دسته‌هایی که واقعاً مدل/پیش‌بینی معتبر دارند (نه هر دوی '0' و '1' کورکورانه)
    # 🆕 run_honesty_check حالا یک تابع سطح ماژول است (بالای فایل، بعد از combine_daughter_and_trend)
    if run_test == 1:
        try:
            categories_to_check = available_categories
        except NameError:
            categories_to_check = []

        for cid in categories_to_check:
            mask = (cat_ids == int(cid))
            if np.any(mask):
                try:
                    run_honesty_check(y_real[mask], all_test_preds[mask], f"Cat_{cid}")
                except Exception as e:
                    print(Fore.RED + f"⚠️ خطا در تست صداقت دسته {cid}: {e}. رد شد، بدون کرش.")

# =================================================================================
# اجرای چرخه‌ی Optuna
# =================================================================================
if run_optuna == 1:
    print(Fore.MAGENTA + "\n🌟 در حال شروع بهینه‌سازی هوشمند با Optuna 🌟")
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    db_path = os.path.join(MODELS_DIR, "optuna_memory.db")
    storage_url = f"sqlite:///{db_path}"

    study = optuna.create_study(
        direction='maximize', 
        study_name="Phase4_KPI_Optimization",
        storage=storage_url, 
        load_if_exists=True
    )
    
    # ➕➕➕ همگام‌سازی glob_best_kpi با دیتابیس Optuna ➕➕➕
    # 🐛 [FIX #5] باگ قبلی: `study.completed_trials` اصلاً در کلاس Study آبجه
    # Optuna وجود ندارد (نه یک attribute واقعی است، نه مستند شده - لیست کامل
    # attribute های Study فقط شامل best_params/best_trial/best_trials/
    # best_value/direction/directions/metric_names/system_attrs/trials/
    # user_attrs است). در نتیجه هر بار اسکریپت ری‌استارت می‌شد، این خط
    # AttributeError می‌داد که توسط `except Exception` گسترده‌ی زیرش بی‌صدا
    # قورت داده می‌شد و glob_best_kpi همیشه به -999999.0 ریست می‌شد - یعنی
    # مقایسه‌ی "بهترین در کل تاریخچه" عملاً هیچ‌وقت واقعی نبود و هر Trial اول
    # بعد از هر ری‌استارت به‌اشتباه به‌عنوان رکورد جدید ثبت و بکاپ می‌گرفت،
    # حتی اگر KPI اش از رکوردهای قبلی (در سشن‌های قبل) کمتر بود.
    # ✅ فیکس: مستقیماً از study.best_value استفاده می‌شود؛ طبق مستندات
    # Optuna، این attribute دقیقاً وقتی هیچ Trial کامل‌شده‌ای وجود نداشته
    # باشد ValueError می‌دهد (نه یک خطای مبهم) - پس فقط همین خطای مشخص گرفته
    # می‌شود، نه هر خطای احتمالی دیگری که ممکن است یک باگ واقعی را هم پنهان کند.
    try:
        glob_best_kpi = study.best_value
        print(Fore.CYAN + f"🔄 بهترین KPI قبلی از Optuna لود شد: {glob_best_kpi:.4f}")
    except ValueError:
        # هیچ Trial کامل‌شده‌ای در دیتابیس نیست (اولین اجرای این study) - شروع از صفر
        glob_best_kpi = -999999.0
        print(Fore.YELLOW + "ℹ️ هیچ Trial کامل‌شده‌ای در دیتابیس پیدا نشد؛ glob_best_kpi از -999999.0 شروع می‌شود.")
    # -------------------------------------------------------------------------

    study.optimize(objective, n_trials=input_n_trials, gc_after_trial=True, catch=(Exception,))
    
    print(Fore.GREEN + "\n🎉 بهینه‌سازی با موفقیت به پایان رسید!")
    print(Fore.CYAN + "بهترین پارامترهای کشف شده برای بالاترین KPI:")
    
    for key, value in study.best_trial.params.items():
        print(Fore.WHITE + f"   👉 {key}: {value}")
        
    print(Fore.YELLOW + f"🏆 بالاترین KPI ثبت شده: {study.best_value}")

    # =========================================================================
    # 🆕 [VIZ #7] نمودارهای تحلیلی خودِ Optuna (بعد از پایان کامل ۳۰۰ Trial):
    #   1) Param Importance -> کدام هایپرپارامتر واقعاً روی KPI تاثیر داشته
    #      (مثلاً اگر TOP_K یا cat_learning_rate عملاً بی‌اثر بودند، دفعه‌ی
    #      بعد می‌شود search space را باریک‌تر کرد و Trial های بیشتری صرف
    #      پارامترهای مهم کرد).
    #   2) Optimization History -> روند پیشرفت KPI در طول ۳۰۰ Trial (آیا هنوز
    #      داشت رشد می‌کرد یا از یک جایی به بعد Plateau شده بود).
    # این بخش نیاز به کتابخانه‌ی plotly دارد (و برای خروجی jpg، کتابخانه‌ی
    # kaleido)؛ اگر نصب نباشند، بدون کرش کل برنامه، فقط یک هشدار چاپ می‌شود و
    # به‌جای jpg، نسخه‌ی HTML تعاملی (که نیازی به kaleido ندارد) ذخیره می‌شود.
    # =========================================================================
    try:
        import optuna.visualization as optuna_vis

        def _save_optuna_fig(fig, base_name):
            try:
                fig.write_image(os.path.join(MODELS_DIR, f"{base_name}.jpg"), width=1100, height=650, scale=2)
                print(Fore.GREEN + f"✅ {base_name}.jpg ذخیره شد.")
            except Exception:
                # 🛡️ اگر kaleido نصب نبود، به‌جای کرش، حداقل نسخه‌ی HTML تعاملی ذخیره می‌شود
                fig.write_html(os.path.join(MODELS_DIR, f"{base_name}.html"))
                print(Fore.YELLOW + f"⚠️ ذخیره‌ی {base_name}.jpg ممکن نشد (احتمالاً kaleido نصب نیست: pip install kaleido). به‌جایش {base_name}.html ذخیره شد.")

        fig_importance = optuna_vis.plot_param_importances(study)
        _save_optuna_fig(fig_importance, "Optuna_Param_Importance")

        fig_history = optuna_vis.plot_optimization_history(study)
        _save_optuna_fig(fig_history, "Optuna_Optimization_History")

    except ImportError:
        print(Fore.YELLOW + "⚠️ کتابخانه‌ی plotly نصب نیست (pip install plotly) - نمودارهای تحلیلی Optuna رد شدند.")
    except Exception as e:
        print(Fore.RED + f"⚠️ خطا در رسم نمودارهای تحلیلی Optuna: {e}. رد شد، بدون کرش.")

    best_params_file = os.path.join(MODELS_DIR, "Optuna_Best_Hyperparameters.txt")
    
    with open(best_params_file, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("   GEMINI ALPHA - FINAL OPTIMIZED HYPERPARAMETERS\n")
        f.write(f"   Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("============================================================\n\n")
        
        f.write(f"🏆 Best KPI (GGI) achieved: {study.best_value:.6f}\n")
        f.write(f"🔢 Total Trials completed: {len(study.trials)}\n\n")
        
        f.write("--- List of Optimized Variables ---\n")
        
        for param_name, param_value in study.best_params.items():
            f.write(f"👉 {param_name:<20} : {param_value}\n")
            
        f.write("\n============================================================")

    print(Fore.GREEN + f"✅ لیست کامل متغیرهای بهینه در فایل ذخیره شد: {best_params_file}")

    # =================================================================================
    # 🆕 آموزش نهایی خودکار با بهترین مقادیر (Auto Final Retrain with Best Params)
    # =================================================================================
    print(Fore.MAGENTA + "\n🔁 در حال بازآموزی نهایی با بهترین مقادیر کشف‌شده (Final Retrain)...")
    print(Fore.MAGENTA + "   این مرحله تضمین می‌کند مدل‌های ذخیره‌شده روی دیسک، دقیقاً مال بهترین Trial باشند.")

    # 🔒 از همین‌جا به بعد Phase 4 دیگه روی Validation (که ۳۰۰ بار دیده شده) حساب
    # نمی‌کند، بلکه روی Final-Test کاملاً دست‌نخورده حساب می‌کند - برای گزارش
    # صادقانه‌ی عملکرد واقعی مدل نهایی، جدا از KPI‌های طول جستجوی Optuna.
    USE_FINAL_HOLDOUT_TEST = True

    study.enqueue_trial(study.best_params)
    study.optimize(objective, n_trials=1, gc_after_trial=True, catch=(Exception,))

    final_trial = study.trials[-1]
    if final_trial.value is None:
        print(Fore.RED + f"⚠️ بازآموزی نهایی (Trial {final_trial.number}) با خطا Fail شد. مدل رسمی قبلی در MODELS_DIR دست‌نخورده باقی می‌ماند.")
        sys.exit()
    print(Fore.GREEN + f"\n✅ آموزش نهایی (انتخاب رسمی Optuna، بر اساس Validation) تمام شد.")


    print(Fore.GREEN + f"   📊 KPI روی Validation (طول جستجو، غیرقابل اعتماد کامل): {glob_best_kpi:.6f}")
    print(Fore.GREEN + f"   🔒 KPI روی Final-Test دست‌نخورده (تخمین صادقانه‌ی واقعی): {final_trial.value:.6f}")

    # 🆕 [SHADOW EVAL] بکاپ صریح از نسخه‌ی رسمی Optuna (بر اساس Val) - قبل از
    # هرگونه بازآموزی احتمالی دوم، تا این نسخه هیچ‌وقت گم نشود، صرف‌نظر از
    # این‌که در نهایت کدام نسخه در MODELS_DIR باقی می‌ماند.
    official_backup_path = f"{MODELS_DIR}_OFFICIAL_OptunaBest_Trial{study.best_trial.number}"
    try:
        shutil.copytree(MODELS_DIR, official_backup_path, dirs_exist_ok=True)
        print(Fore.GREEN + f"   💾 نسخه‌ی رسمی (انتخاب Optuna بر اساس Validation) بکاپ گرفته شد:\n      {official_backup_path}")
    except Exception as e:
        print(Fore.RED + f"   ⚠️ خطا در بکاپ‌گیری از نسخه‌ی رسمی: {e}")

    # =================================================================================
    # 🆕 [SHADOW EVAL] آیا یک Trial دیگر واقعاً روی Holdout بهتر بوده؟
    # اگر بله، طبق فلگ PREFER_HOLDOUT_OVER_VAL_FOR_FINAL_MODEL تصمیم می‌گیریم
    # که مدل نهایی در MODELS_DIR (همانی که Phase 5/Live از آن می‌خواند) بر
    # اساس کدام Trial باشد. هر دو نسخه، صرف‌نظر از این تصمیم، جداگانه و
    # واضح روی دسکتاپ بکاپ گرفته می‌شوند - هیچ اطلاعاتی گم نمی‌شود، فقط
    # این‌که کدام یک در مسیر "زنده" (MODELS_DIR) قرار می‌گیرد فرق می‌کند.
    # =================================================================================
    print(Fore.MAGENTA + "\n🔎 در حال بررسی Shadow Holdout Log برای یافتن واقعاً بهترین Trial...")
    true_best_trial_num, true_best_holdout_kpi, true_best_val_kpi = get_true_best_trial_from_shadow_log()

    if true_best_trial_num is None:
        print(Fore.YELLOW + "   ℹ️ داده‌ی کافی در Shadow_Holdout_Log.csv نبود (یا این قابلیت جدید اضافه شده و تاریخچه‌ی قدیمی را پوشش نمی‌دهد).")
        print(Fore.YELLOW + "   مدل رسمی Optuna (بر اساس Validation) به‌عنوان مدل نهایی در MODELS_DIR باقی می‌ماند.")
    elif true_best_holdout_kpi is not None and true_best_holdout_kpi <= 0:
        print(Fore.RED + f"   🚨 هشدار جدی: حتی بهترین Trial از نظر Holdout هم (#{true_best_trial_num}) KPI نامثبت داشته ({true_best_holdout_kpi:.4f}).")
        print(Fore.RED + f"      یعنی در کل این دور جست‌وجو، هیچ Trial ای روی داده‌ی واقعاً ندیده سودآور نبوده.")
        print(Fore.RED + f"      مدل رسمی Optuna در MODELS_DIR باقی می‌ماند، ولی توصیه می‌شود قبل از Live از آن استفاده نکنید.")
    elif true_best_trial_num == study.best_trial.number:
        print(Fore.GREEN + f"   ✅ انتخاب رسمی Optuna (Trial #{study.best_trial.number}) دقیقاً همان Trial ای است که روی Holdout هم بهترین بوده.")
        print(Fore.GREEN + f"      نیازی به بازآموزی دوم نیست؛ می‌توانی با اطمینان بیشتری به مدل فعلی در MODELS_DIR اعتماد کنی.")
    else:
        print(Fore.YELLOW + f"   ⚠️ تفاوت پیدا شد: انتخاب رسمی Optuna Trial #{study.best_trial.number} است (Val={study.best_value:.4f}),")
        print(Fore.YELLOW + f"      ولی Trial #{true_best_trial_num} روی Holdout واقعی بهتر بوده (Val={true_best_val_kpi:.4f}, Holdout={true_best_holdout_kpi:.4f}).")

        if PREFER_HOLDOUT_OVER_VAL_FOR_FINAL_MODEL:
            print(Fore.MAGENTA + f"   🔁 PREFER_HOLDOUT_OVER_VAL_FOR_FINAL_MODEL=True است؛ در حال بازآموزی دوم با پارامترهای Trial #{true_best_trial_num}...")
            true_best_trial_obj = next((t for t in study.trials if t.number == true_best_trial_num), None)
            if true_best_trial_obj is None:
                print(Fore.RED + f"      ⚠️ پارامترهای Trial #{true_best_trial_num} در دیتابیس Optuna پیدا نشد (احتمالاً از یک اجرای قدیمی‌تر است). مدل رسمی دست‌نخورده باقی می‌ماند.")
            else:
                study.enqueue_trial(true_best_trial_obj.params)
                study.optimize(objective, n_trials=1, gc_after_trial=True, catch=(Exception,))
                true_best_retrain_trial = study.trials[-1]
                print(Fore.GREEN + f"   ✅ بازآموزی دوم تمام شد. مدل نهایی در MODELS_DIR حالا بر اساس Trial #{true_best_trial_num} (واقعاً بهترین روی Holdout) است.")
                print(Fore.GREEN + f"      🔒 KPI روی Final-Test در این بازآموزی: {true_best_retrain_trial.value:.6f}")

                truebest_backup_path = f"{MODELS_DIR}_TrueBest_Holdout_Trial{true_best_trial_num}"
                try:
                    shutil.copytree(MODELS_DIR, truebest_backup_path, dirs_exist_ok=True)
                    print(Fore.GREEN + f"      💾 این نسخه هم جداگانه بکاپ گرفته شد:\n         {truebest_backup_path}")
                except Exception as e:
                    print(Fore.RED + f"      ⚠️ خطا در بکاپ‌گیری از نسخه‌ی TrueBest: {e}")
        else:
            print(Fore.YELLOW + f"   ℹ️ PREFER_HOLDOUT_OVER_VAL_FOR_FINAL_MODEL=False است؛ مدل رسمی Optuna در MODELS_DIR بدون تغییر باقی می‌ماند.")
            print(Fore.YELLOW + f"      اگر می‌خواهی به‌جایش از Trial #{true_best_trial_num} استفاده کنی، بکاپ آن را از پوشه‌ی BEST_Trial{true_best_trial_num}_* روی دسکتاپ پیدا کن (اگر موجود بود) یا این فلگ را True کن و دوباره اجرا کن.")

    print(Fore.GREEN + f"\n   مدل‌های نهایی (mother_model.keras / model_0.keras / model_1.keras / trend_model.keras / meta_model_*.pkl)")
    print(Fore.GREEN + f"   در مسیر زیر ذخیره شدند و آماده‌ی استفاده در Phase 5 (Live) هستند:")
    print(Fore.GREEN + f"   {MODELS_DIR}")

    sys.exit()

elif train_mother == 1 or train_daughter in [0, 1, 2] or th_choice in [0, 1, 2] or run_test == 1:
    # 🆕 [FIX #4] حالت دستی (بدون Optuna): این شاخه دقیقاً همان چیزی را اجرا
    # می‌کند که در منوی «SYSTEM CONFIGURATION» انتخاب کرده‌ای
    # (train_mother / train_daughter / th_choice / run_test) - یک‌بار، با
    # هایپرپارامترهای فعلی (مقادیر پیش‌فرض بالای فایل، نه trial.suggest_*).
    # مثلاً اگر فقط th_choice=2 زده باشی و بقیه را «Load Existing»، فقط فاز ۳
    # اجرا می‌شود و مادر/دختر از دیسک لود می‌شوند - هر فاز کاملاً مستقل.
    print(Fore.MAGENTA + "\n🔧 در حال اجرای دستی بر اساس انتخاب‌های منو (بدون Optuna)...")
    objective(trial=None)
else:
    print(Fore.WHITE + "\nℹ️ هیچ‌کدام از فازهای ۱ تا ۴ انتخاب نشدند؛ مستقیم می‌رویم سراغ فاز ۵ (بر اساس LIVE_MODE).")


# --- PHASE 5: LIVE TRADING BRIDGE (Optimized for 2-Model System + Trend + Meta) ---   ---------------------------------      PHASE 5


#   تابع پیدا کردن پورت خالی
def find_free_port(default_port):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('0.0.0.0', default_port))
        s.close()
        return default_port
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp_s:
            temp_s.bind(('', 0))
            new_port = temp_s.getsockname()[1]
            return new_port
        


if LIVE_MODE:
    print(Fore.MAGENTA + "\n=== PHASE 5: LIVE TRADING BRIDGE (URL Server Active) ===")

    app = Flask(__name__)

    # ۱. تست سلامت فایل‌های حیاتی (Pre-flight Check)
    # 🛡️ اصلاح: به‌جای نیازمند بودن HARDCODE هر دو دسته (SELL+BUY)، این‌جا فقط
    # فایل‌های مشترک (فیچرهای مادر) الزامی هستند. هر دسته (SELL/BUY) جداگانه
    # چک می‌شود و اگر مدل آن دسته موجود نبود، فقط آن دسته غیرفعال می‌شود
    # (به‌جای متوقف کردن کل سرور با exit()).
    print(Fore.CYAN + "🔍 در حال بررسی مدارک، مدل‌ها و تریشلدهای سیستم...")

    core_required_files = [
        os.path.join(MODELS_DIR, "union_features_mother.pkl"),
    ]
    missing_core = [f for f in core_required_files if not os.path.exists(f)]
    if missing_core:
        print(Fore.RED + "❌ خطا: فایل‌های پایه‌ای زیر پیدا نشدند. ابتدا فاز ۱ تا ۳ را اجرا کنید:")
        for mf in missing_core:
            print(Fore.YELLOW + f"   - {os.path.basename(mf)}")
        exit()
    else:
        print(Fore.GREEN + "✅ فایل‌های پایه‌ای تایید شدند.")

    # ۲. آماده‌سازی دارایی‌ها (مدل‌ها، اسکیلرها و تریشلدها به صورت داینامیک)
    try:
        loaded_assets = {}
        union_feat_path = os.path.join(MODELS_DIR, "union_features_mother.pkl")
        current_feat_names = joblib.load(union_feat_path)
        
        df_dummy = pd.read_csv(DATASET_PATH, sep=';', nrows=1)
        df_dummy.columns = [str(c).strip() for c in df_dummy.columns]
        
        current_indices = [df_dummy.columns.get_loc(name) for name in current_feat_names]
        model_specific_top_k = len(current_feat_names) // TIME_STEPS

        for cat_id in ['0', '1']:
            m_path = os.path.join(CAT_MODELS_PATH, f"model_{cat_id}.keras")
            s_path = os.path.join(SCALERS_PATH, f"scaler_{cat_id}.pkl")
            cal_path = os.path.join(MODELS_DIR, f"calibrator_{cat_id}.pkl")

            # 🛡️ اگر مدل این دسته (مثلاً SELL) هنوز آموزش داده نشده، فقط این دسته را
            # رد می‌کنیم و پیام می‌دهیم - بدون متوقف کردن کل سرور Live.
            if not (os.path.exists(m_path) and os.path.exists(s_path)):
                cat_name = "SELL" if cat_id == '0' else "BUY"
                print(Fore.YELLOW + f"⚠️ مدل/اسکیلر دسته {cat_name} (Cat {cat_id}) پیدا نشد. این دسته در حالت Live غیرفعال می‌ماند (فقط دسته‌های دیگر پاسخ می‌دهند).")
                continue

            th_file_name = "th_sell_0.pkl" if cat_id == '0' else "th_buy_1.pkl"
            th_full_path = os.path.join(MODELS_DIR, th_file_name)
            current_th = float(joblib.load(th_full_path)) if os.path.exists(th_full_path) else 0.5

            loaded_assets[cat_id] = {
                'model': tf.keras.models.load_model(m_path, custom_objects={'MultiHeadSelfAttention': MultiHeadSelfAttention}),
                'scaler': joblib.load(s_path),
                'calibrator': joblib.load(cal_path) if os.path.exists(cal_path) else None,
                'th': current_th
            }
            print(Fore.GREEN + f"✔️ Cat {cat_id} Ready | TH: {current_th}")

        if len(loaded_assets) == 0:
            print(Fore.RED + "❌ هیچ مدل دسته‌ای (نه SELL نه BUY) در دسترس نیست. سرور Live متوقف می‌شود.")
            exit()

        # --- 🆕 آماده‌سازی مدل روند (اختیاری) برای ترکیب زنده با متا-مدل ---
        trend_model_file = os.path.join(TREND_MODEL_PATH, "trend_model.keras")
        trend_scaler_file = os.path.join(TREND_MODEL_PATH, "trend_scaler.pkl")
        trend_feats_file = os.path.join(TREND_MODEL_PATH, "trend_features.pkl")
        TREND_LIVE_READY = os.path.exists(trend_model_file) and os.path.exists(trend_scaler_file) and os.path.exists(trend_feats_file)

        if TREND_LIVE_READY:
            trend_live_model = tf.keras.models.load_model(trend_model_file, custom_objects={'MultiHeadSelfAttention': MultiHeadSelfAttention})
            trend_live_scaler = joblib.load(trend_scaler_file)
            trend_live_feat_names = joblib.load(trend_feats_file)
            # پیدا کردن ایندکس ستون‌های روند در همان raw_string ورودی زنده (اگر موجود باشند)
            try:
                trend_live_indices = [df_dummy.columns.get_loc(name) for name in trend_live_feat_names]
            except KeyError:
                print(Fore.YELLOW + "⚠️ ستون‌های روند در دیتاست فعلی پیدا نشدند (EA هنوز آپدیت نشده). مدل روند در Live غیرفعال می‌ماند.")
                TREND_LIVE_READY = False
                trend_live_indices = []
            trend_live_top_k = len(trend_live_feat_names) // TIME_STEPS if TREND_LIVE_READY else 0
            # 🆕 [FIX #1] لود کالیبراتور مدل روند (اگر موجود باشد) - چون مسیر Live از
            # predict_trend_probability() استفاده نمی‌کند و پیش‌بینی روند را این‌جا
            # مستقیماً و جداگانه انجام می‌دهد، باید همین‌جا هم کالیبراتور را جدا لود کنیم.
            trend_live_calibrator_path = os.path.join(TREND_MODEL_PATH, "trend_calibrator.pkl")
            trend_live_calibrator = joblib.load(trend_live_calibrator_path) if os.path.exists(trend_live_calibrator_path) else None
            if TREND_LIVE_READY:
                cal_status = "با کالیبراتور" if trend_live_calibrator is not None else "بدون کالیبراتور (خام)"
                print(Fore.GREEN + f"✔️ Trend Model Ready | Families: {trend_live_top_k} | {cal_status}")
        else:
            print(Fore.YELLOW + "⚠️ مدل روند در دسترس نیست. سیستم فقط با خروجی خام مدل‌های دختر کار می‌کند (بدون متا-مدل).")
            trend_live_indices = []
            trend_live_top_k = 0
            trend_live_calibrator = None

        # لود متا-مدل‌های موجود (اختیاری، هر دسته جدا)
        meta_models_live = {}
        for cat_id in loaded_assets.keys():
            meta_p = os.path.join(META_MODELS_PATH, f"meta_model_{cat_id}.pkl")
            if os.path.exists(meta_p):
                meta_models_live[cat_id] = joblib.load(meta_p)
                print(Fore.GREEN + f"✔️ Meta-Model Cat {cat_id} Ready (Daughter + Trend).")

        SHARED_CONTEXT = {
            'indices': current_indices,
            'top_k': model_specific_top_k
        }

        print(Fore.MAGENTA + "🚀 سرور آماده دریافت سیگنال داینامیک است...")
    except Exception as e:
        print(Fore.RED + f"❌ خطای بحرانی در لودینگ دارایی‌ها: {e}")
        exit()

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            content = request.get_json(force=True, silent=True)
            if content is None or 'data' not in content:
                return jsonify({'command': 'ERROR', 'msg': 'No data'})

            raw_string = content.get('data', '')
            data_parts = raw_string.split(';')
            
            if len(data_parts) >= 1:
                c_short = str(int(float(data_parts[0]))) 

                # 🛡️ اگر مدل این دسته (مثلاً SELL) اصلاً لود نشده، به‌جای کرش، پاسخ WAIT بده
                if c_short not in loaded_assets:
                    return jsonify({'command': 'WAIT', 'msg': f'Model for category {c_short} not available'})

                assets = loaded_assets[c_short]

                # استخراج فیچرهای مدل دختر (فوق سریع)
                selected_values = [float(data_parts[idx]) for idx in SHARED_CONTEXT['indices']]
                selected_np = np.array(selected_values).reshape(1, -1)

                X_scaled = assets['scaler'].transform(selected_np)
                X_3d = X_scaled.reshape(1, SHARED_CONTEXT['top_k'], TIME_STEPS)
                X_3d = np.transpose(X_3d, (0, 2, 1))

                raw_prediction = assets['model'](X_3d, training=False).numpy()[0][0]

                if assets['calibrator'] is not None:
                    prediction = float(assets['calibrator'].predict_proba(np.array([[raw_prediction]]))[:, 1][0])
                else:
                    prediction = raw_prediction

                # --- 🆕 پیش‌بینی روند + ترکیب با متا-مدل (اگر هر دو موجود باشند) ---
                used_meta_live = False
                if TREND_LIVE_READY and c_short in meta_models_live:
                    try:
                        trend_values = [float(data_parts[idx]) for idx in trend_live_indices]
                        trend_np = np.array(trend_values).reshape(1, -1)
                        trend_scaled = trend_live_scaler.transform(trend_np)
                        trend_3d = trend_scaled.reshape(1, trend_live_top_k, TIME_STEPS)
                        trend_3d = np.transpose(trend_3d, (0, 2, 1))
                        trend_prediction = trend_live_model(trend_3d, training=False).numpy()[0][0]

                        # 🆕 [FIX #1] اعمال کالیبراتور روند (اگر موجود باشد) - قبل از ورود
                        # به متا-مدل، دقیقاً هم‌مقیاس با چیزی که متا-مدل رویش آموزش دیده.
                        if trend_live_calibrator is not None:
                            trend_prediction = float(trend_live_calibrator.predict_proba(np.array([[trend_prediction]]))[:, 1][0])

                        meta_model_live = meta_models_live[c_short]
                        meta_input = build_meta_features([prediction], [trend_prediction])
                        prediction = float(meta_model_live.predict_proba(meta_input)[:, 1][0])
                        used_meta_live = True
                    except Exception as e:
                        print(Fore.YELLOW + f"⚠️ خطا در ترکیب زنده‌ی مدل روند/متا: {e}. فقط خروجی خام مدل دختر استفاده شد.")

                th = assets['th']
                now_time = datetime.now().strftime("%H:%M:%S")

                if c_short == "1": # BUY Model
                    final_command = "BUY" if prediction >= th else "NOBUY"
                    color = Fore.GREEN if "NO" not in final_command else Fore.YELLOW
                elif c_short == "0": # SELL Model
                    final_command = "SELL" if prediction >= th else "NOSELL"
                    color = Fore.RED if "NO" not in final_command else Fore.YELLOW
                else:
                    final_command = "WAIT"
                    color = Fore.WHITE

                meta_tag = " [Meta]" if used_meta_live else ""
                print(f"[{now_time}] Cat: {c_short} | Prob: {prediction:.4f}{meta_tag} | TH: {th} | Result: {color}{final_command}")
                return jsonify({'command': final_command, 'prob': prediction, 'meta_used': used_meta_live})
                
            return jsonify({'command': 'WAIT'})

        except Exception as e:
            print(Fore.RED + f"Error in Predict: {e}")
            return jsonify({'command': 'ERROR', 'msg': str(e)})

    # --- این بخش باید بیرون از تابع باشد (هم‌سطح با @app.route) ---
    DEFAULT_PORT = 8000
    active_port = find_free_port(DEFAULT_PORT)
    
    print("\n" + "="*50)
    print(Fore.MAGENTA + f"📡 AI Server is starting...")
    if active_port == DEFAULT_PORT:
        print(Fore.GREEN + f"✅ Using Default Port: {active_port}")
    else:
        print(Fore.RED + f"⚠️ Port {DEFAULT_PORT} was BUSY!")
        print(Fore.YELLOW + f"👉 NEW ACTIVE PORT: {active_port}")
        print(Fore.CYAN + "Please update the URL in your Expert Advisor (MT4).")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=active_port, debug=False, use_reloader=False)