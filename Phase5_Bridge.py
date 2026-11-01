"""Standalone Phase 5 bridge. Models stay outside the executable."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
import sys, json, argparse, logging, threading, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from flask import Flask, request, jsonify
TIME_STEPS=30
MODEL_FOLDER='BEST_Trial257_KPI_0.0075_2026-10-07_23-31-58'
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
    



def home():
    return Path(sys.executable if getattr(sys,'frozen',False) else __file__).resolve().parent

def resource(name):
    p=home()/name
    return p if p.exists() else Path(getattr(sys,'_MEIPASS',home()))/name

def locate_models(override=None):
    base=Path(override).resolve() if override else home()/MODEL_FOLDER
    for p in (base,base/MODEL_FOLDER,home()):
        if (p/'union_features_mother.pkl').is_file(): return p
    raise FileNotFoundError('Place '+MODEL_FOLDER+' beside this program, or use --models PATH')

class Predictor:
    def __init__(self,model,scaler,features):
        self.model=model; self.scaler=scaler; self.features=features
        shape=(1,TIME_STEPS,len(features)//TIME_STEPS)
        if len(features)%TIME_STEPS or tuple(model.input_shape[1:])!=shape[1:]:
            raise ValueError('Model and feature dimensions do not match')
        self.call=tf.function(lambda x:model(x,training=False),input_signature=[tf.TensorSpec(shape,tf.float32)],autograph=False)
        self.call(tf.zeros(shape,tf.float32)).numpy() # Build graph before serving any requests.
    def predict(self,values):
        x=np.asarray(values,dtype=np.float64).reshape(1,-1)
        if not np.isfinite(x).all(): raise ValueError('Non-finite feature value')
        if hasattr(self.scaler,'feature_names_in_'):
            x=pd.DataFrame(x,columns=self.scaler.feature_names_in_)
        x=self.scaler.transform(x).reshape(1,-1,TIME_STEPS).transpose(0,2,1)
        return float(self.call(tf.convert_to_tensor(x,dtype=tf.float32)).numpy().ravel()[0])

class Bridge:
    def __init__(self,folder):
        self.folder=folder; self.lock=threading.Lock(); self.assets={}
        self.columns=json.loads(resource('phase5_columns.json').read_text(encoding='utf-8'))
        def load(name): return joblib.load(folder/name)
        def network(name):
            return tf.keras.models.load_model(folder/name,custom_objects={'MultiHeadSelfAttention':MultiHeadSelfAttention},compile=False)
        self.features=load('union_features_mother.pkl')
        self.indices=[self.columns.index(n) for n in self.features]
        for cat in ('0','1'):
            if not (folder/f'Category_Models/model_{cat}.keras').exists() or not (folder/f'Scalers/scaler_{cat}.pkl').exists():
                print(f'Category {cat} unavailable: requests return WAIT.',flush=True); continue
            self.assets[cat]={
                'predictor':Predictor(network(f'Category_Models/model_{cat}.keras'),load(f'Scalers/scaler_{cat}.pkl'),self.features),
                'calibrator':load(f'calibrator_{cat}.pkl'),
                'meta':load(f'Meta_Models/meta_model_{cat}.pkl'),
                'th':float(load('th_buy_1.pkl' if cat=='1' else 'th_sell_0.pkl'))}
            if not 0<=self.assets[cat]['th']<=1: raise ValueError('Invalid threshold')
            print(f'Category {cat} READY | threshold={self.assets[cat]["th"]}',flush=True)
        if not self.assets: raise RuntimeError('No category models found')
        self.trend_features=load('Trend_Model/trend_features.pkl')
        self.trend_indices=[self.columns.index(n) for n in self.trend_features]
        self.trend=Predictor(network('Trend_Model/trend_model.keras'),load('Trend_Model/trend_scaler.pkl'),self.trend_features)
        self.trend_calibrator=load('Trend_Model/trend_calibrator.pkl')
    def predict(self,data):
        start=time.perf_counter()
        if not isinstance(data,str): raise ValueError('data must be a semicolon-separated string')
        parts=data.strip().rstrip(';').split(';')
        category=float(parts[0])
        if category not in (0,1): raise ValueError('Category must be 0 or 1')
        cat=str(int(category))
        if cat not in self.assets: return {'command':'WAIT','msg':'Model for this category not available'}
        def values(indices):
            if len(parts)<=max(indices): raise ValueError('Incomplete feature row')
            return [float(parts[i]) for i in indices]
        a=self.assets[cat]
        with self.lock:
            raw=a['predictor'].predict(values(self.indices))
            p=float(a['calibrator'].predict_proba([[raw]])[0,1])
            tr=self.trend.predict(values(self.trend_indices))
            t=float(self.trend_calibrator.predict_proba([[tr]])[0,1])
            dp=np.asarray([p],dtype='float32'); tp=np.asarray([t],dtype='float32')
            meta=np.column_stack([dp,tp,dp*tp,np.abs(dp-tp)])
            final=float(a['meta'].predict_proba(meta)[0,1])
        if not np.isfinite(final) or not 0<=final<=1: raise ValueError('Invalid probability')
        command=('BUY' if cat=='1' else 'SELL') if final>=a['th'] else ('NOBUY' if cat=='1' else 'NOSELL')
        elapsed=(time.perf_counter()-start)*1000
        print(f'Cat={cat} Prob={final:.6f} TH={a["th"]} Result={command} | inference={elapsed:.1f} ms',flush=True)
        return {'command':command,'prob':final,'meta_used':True,'inference_ms':round(elapsed,2)}

def make_app(bridge):
    app=Flask(__name__); app.config['MAX_CONTENT_LENGTH']=1024*1024
    @app.get('/health')
    def health(): return jsonify(status='ready',categories=list(bridge.assets),models=str(bridge.folder),optimized=True)
    @app.post('/predict')
    def predict():
        try:
            body=request.get_json(silent=True)
            if not isinstance(body,dict) or 'data' not in body: raise ValueError('No data')
            return jsonify(bridge.predict(body['data']))
        except Exception as e:
            logging.exception('Prediction rejected')
            return jsonify(command='ERROR',msg=str(e))
    return app

def main():
    parser=argparse.ArgumentParser(description='Optimized standalone Phase 5 MT4 bridge')
    parser.add_argument('--models'); parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    logging.basicConfig(filename=home()/'Phase5_Bridge.log',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    print('Loading and warming optimized Phase 5 models. Please wait...',flush=True)
    b=Bridge(locate_models(args.models))
    if args.check: print('MODEL CHECK PASSED',flush=True); return
    print(f'READY: http://127.0.0.1:{args.port}/predict | Optimized | Close window to stop.',flush=True)
    make_app(b).run(host='127.0.0.1',port=args.port,debug=False,use_reloader=False,threaded=True)

if __name__=='__main__':
    try: main()
    except (Exception,SystemExit) as e:
        if isinstance(e,SystemExit) and e.code in (None,0): raise
        logging.exception('Startup failed'); print(f'STARTUP FAILED: {e}',flush=True)
        if getattr(sys,'frozen',False) and sys.stdin and sys.stdin.isatty(): input('Press Enter to close...')
        sys.exit(1)
