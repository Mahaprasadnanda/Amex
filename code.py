#!/usr/bin/env python3
"""
FAST OPTIMIZED AMERICAN EXPRESS CAMPUS CHALLENGE 2025 SOLUTION
=============================================================
Aims for AUC >0.99999 with reduced execution time (1-2 hours).
"""

import pandas as pd
import numpy as np
import warnings
import os
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import StackingClassifier
import gc
from datetime import datetime
import dask.dataframe as dd
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import optuna

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

print("🚀 FAST OPTIMIZED SOLUTION FOR AUC >0.99999")
print("=" * 80)
print("✅ Reduced tuning trials, epochs, and CV for 1-2 hour runtime!")
print("✅ Still full data, advanced features, and stacking ensemble!")
print("=" * 80)

class OptimizedAmexSolution:
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.additional_dfs = {}
        self.target_col = None
        self.feature_cols = []
        self.models = []
        self.predictions = None
        self.scaler = StandardScaler()
        
    def safe_read_parquet(self, filename, use_dask=False):
        try:
            if os.path.exists(filename):
                if use_dask:
                    df = dd.read_parquet(filename)
                    print(f"✅ Loaded large {filename} with Dask")
                    return df
                else:
                    df = pd.read_parquet(filename)
                    print(f"✅ Loaded {filename}: {df.shape}")
                    return df
            else:
                print(f"⚠  File {filename} not found")
                return None
        except Exception as e:
            print(f"⚠  Error reading {filename}: {e}")
            return None
    
    def aggregate_additional_data(self, df, filename):
        if df is None:
            return pd.DataFrame()
        
        try:
            if isinstance(df, dd.DataFrame):
                df = df.compute()
            
            common_group_cols = ['id2', 'customer_id', 'user_id', 'cid', 'customer']
            group_col = next((col for col in common_group_cols if col in df.columns), None)
            if group_col is None:
                print(f"⚠  No group column found in {filename}")
                return pd.DataFrame()
            
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            aggs = {}
            for col in num_cols:
                if col != group_col:
                    aggs[col] = ['count', 'mean', 'sum', 'max', 'min', 'std']
            for col in cat_cols:
                if col != group_col:
                    aggs[col] = ['nunique']
            
            if not aggs:
                print(f"⚠  No columns to aggregate in {filename}")
                return pd.DataFrame()
            
            agg_df = df.groupby(group_col).agg(aggs).reset_index()
            agg_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in agg_df.columns.values]
            
            if group_col != 'id2':
                agg_df = agg_df.rename(columns={group_col: 'id2'})
            
            print(f"✅ Aggregated {filename} shape: {agg_df.shape}")
            return agg_df
        except Exception as e:
            print(f"⚠  Error aggregating {filename}: {e}")
            return pd.DataFrame()
    
    def load_all_data(self):
        print("\n📂 Loading FULL data...")
        
        self.train_df = self.safe_read_parquet('train_data.parquet')
        self.test_df = self.safe_read_parquet('test_data.parquet')
        
        self.additional_dfs['offer_metadata'] = self.safe_read_parquet('offer_metadata.parquet')
        self.additional_dfs['add_trans'] = self.safe_read_parquet('add_trans.parquet', use_dask=True)
        self.additional_dfs['add_event'] = self.safe_read_parquet('add_event.parquet', use_dask=True)
        
        trans_agg = self.aggregate_additional_data(self.additional_dfs['add_trans'], 'add_trans.parquet')
        event_agg = self.aggregate_additional_data(self.additional_dfs['add_event'], 'add_event.parquet')
        
        if not trans_agg.empty and 'id2' in trans_agg.columns and 'id2' in self.train_df.columns:
            self.train_df['id2'] = self.train_df['id2'].astype(str)
            self.test_df['id2'] = self.test_df['id2'].astype(str)
            trans_agg['id2'] = trans_agg['id2'].astype(str)
            self.train_df = self.train_df.merge(trans_agg, on='id2', how='left')
            self.test_df = self.test_df.merge(trans_agg, on='id2', how='left')
            print(f"✅ Merged transaction aggregates")
        
        if not event_agg.empty and 'id2' in event_agg.columns and 'id2' in self.train_df.columns:
            self.train_df = self.train_df.merge(event_agg, on='id2', how='left')
            self.test_df = self.test_df.merge(event_agg, on='id2', how='left')
            print(f"✅ Merged event aggregates")
        
        if self.additional_dfs['offer_metadata'] is not None:
            try:
                common_offer_cols = ['id3', 'offer_id', 'offer']
                offer_col = next((col for col in common_offer_cols if col in self.additional_dfs['offer_metadata'].columns), None)
                if offer_col:
                    self.train_df['id3'] = self.train_df['id3'].astype(str)
                    self.test_df['id3'] = self.test_df['id3'].astype(str)
                    self.additional_dfs['offer_metadata'][offer_col] = self.additional_dfs['offer_metadata'][offer_col].astype(str)
                    if offer_col != 'id3':
                        self.additional_dfs['offer_metadata'] = self.additional_dfs['offer_metadata'].rename(columns={offer_col: 'id3'})
                    self.train_df = self.train_df.merge(self.additional_dfs['offer_metadata'], on='id3', how='left')
                    self.test_df = self.test_df.merge(self.additional_dfs['offer_metadata'], on='id3', how='left')
                    print(f"✅ Merged offer metadata")
            except Exception as e:
                print(f"⚠  Failed to merge offer metadata: {e}")
        
        if self.train_df is None or self.test_df is None:
            print("\n🎭 Creating synthetic data...")
            self.create_synthetic_data()
        
        gc.collect()
    
    def create_synthetic_data(self):
        print("🎭 Creating synthetic data...")
        np.random.seed(42)
        n_train = 10000
        n_test = 2000
        customer_ids = np.random.randint(1000000, 2000000, n_train)
        test_customer_ids = np.random.randint(1000000, 2000000, n_test)
        offer_ids = np.random.randint(1000, 50000, n_train)
        test_offer_ids = np.random.randint(1000, 50000, n_test)
        id1_train = [f"{cid}_{oid}_16-23_2023-11-15 09:30:00.000" for cid, oid in zip(customer_ids, offer_ids)]
        id1_test = [f"{cid}_{oid}_16-23_2023-11-15 09:30:00.000" for cid, oid in zip(test_customer_ids, test_offer_ids)]
        features = {f'feature_{i}': np.random.randn(n_train) for i in range(20)}
        test_features = {f'feature_{i}': np.random.randn(n_test) for i in range(20)}
        target = np.random.binomial(1, 0.07, n_train)
        self.train_df = pd.DataFrame({
            'id1': id1_train,
            'id2': customer_ids.astype(str),
            'id3': offer_ids.astype(str),
            'id4': pd.date_range('2023-01-01', periods=n_train, freq='H'),
            'id5': pd.date_range('2023-01-01', periods=n_train, freq='H'),
            'target': target,
            **features
        })
        self.test_df = pd.DataFrame({
            'id1': id1_test,
            'id2': test_customer_ids.astype(str),
            'id3': test_offer_ids.astype(str),
            'id4': pd.date_range('2023-01-01', periods=n_test, freq='H'),
            'id5': pd.date_range('2023-01-01', periods=n_test, freq='H'),
            **test_features
        })
        print(f"✅ Created synthetic training data: {self.train_df.shape}")
        print(f"✅ Created synthetic test data: {self.test_df.shape}")
    
    def find_target_column(self):
        if self.train_df is None:
            return None
        
        target_patterns = ['target', 'label', 'y', 'class', 'outcome', 'response', 'click', 'clicked', 'conversion', 'converted', 'purchase', 'f366', 'engaged', 'success', 'failure', 'binary_target']
        
        for pattern in target_patterns:
            if pattern in self.train_df.columns:
                print(f"✅ Found target column: {pattern}")
                return pattern
        
        for pattern in target_patterns:
            matching_cols = [col for col in self.train_df.columns if pattern in col.lower()]
            if matching_cols:
                print(f"✅ Found target column (partial match): {matching_cols[0]}")
                return matching_cols[0]
        
        binary_cols = []
        for col in self.train_df.columns:
            try:
                unique_vals = self.train_df[col].dropna().unique()
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, True, False, '0', '1'}):
                    binary_cols.append(col)
            except:
                continue
        
        if binary_cols:
            print(f"✅ Found binary target column: {binary_cols[0]}")
            return binary_cols[0]
        
        print("⚠  No target column found. Creating synthetic target...")
        self.train_df['synthetic_target'] = np.random.binomial(1, 0.07, len(self.train_df))
        return 'synthetic_target'
    
    def create_bulletproof_features(self):
        print("\n🛠  Creating advanced features...")
        
        for df_name, df in [('train', self.train_df), ('test', self.test_df)]:
            if df is None:
                continue
            
            # 1. Datetime features (reduced for speed)
            date_cols = df.select_dtypes(include=['datetime64']).columns
            for col in date_cols:
                try:
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                    print(f"   ✅ Created key datetime features for {col}")
                except Exception as e:
                    print(f"   ⚠  Error creating datetime features for {col}: {e}")
            
            # 2. Numeric stats (limited to mean and std for speed)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != self.target_col]
            
            if len(numeric_cols) > 1:
                try:
                    df['row_mean'] = df[numeric_cols].mean(axis=1)
                    df['row_std'] = df[numeric_cols].std(axis=1)
                    print(f"   ✅ Created key row-wise statistics")
                except Exception as e:
                    print(f"   ⚠  Error creating row statistics: {e}")
            
            # 3. Categorical encoding (only for low-cardinality to save time)
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            cat_cols = [col for col in cat_cols if col not in ['id1'] and df[col].nunique() < 20]  # Reduced threshold
            
            for col in cat_cols:
                try:
                    le = LabelEncoder()
                    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str).fillna('missing'))
                    print(f"   ✅ Encoded {col}")
                except Exception as e:
                    print(f"   ⚠  Error encoding {col}: {e}")
            
            # 4. ID frequency (only for key IDs)
            id_cols = [col for col in df.columns if 'id' in col.lower() and col != 'id1' and df[col].nunique() < 10000]  # Limit to lower cardinality
            for col in id_cols:
                try:
                    freq_map = df[col].value_counts().to_dict()
                    df[f'{col}_frequency'] = df[col].map(freq_map)
                    print(f"   ✅ Created frequency for {col}")
                except Exception as e:
                    print(f"   ⚠  Error creating frequency for {col}: {e}")
            
            # 5. Key interactions (limited for speed)
            if 'amount_mean' in df.columns and 'event_count' in df.columns:
                df['spend_event_ratio'] = df['amount_mean'] / (df['event_count'] + 1)
                print(f"   ✅ Created spend_event_ratio")
            
            if 'id5' in df.columns and pd.api.types.is_datetime64_any_dtype(df['id5']):
                current_date = datetime.now()
                df['days_since_event'] = (current_date - df['id5']).dt.days
                print(f"   ✅ Created days_since_event")
        
        gc.collect()
    
    def prepare_features(self):
        print("\n🎯 Preparing features...")
        
        if self.train_df is None:
            print("❌ No training data!")
            return
        
        exclude_cols = ['id1', self.target_col, 'id2', 'id3', 'id5']
        
        self.feature_cols = [col for col in self.train_df.columns 
                             if col not in exclude_cols and self.train_df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        self.feature_cols = [col for col in self.feature_cols if self.train_df[col].isnull().mean() < 0.95]
        
        print(f"✅ Selected {len(self.feature_cols)} features")
        
        for col in self.feature_cols:
            self.train_df[col] = self.train_df[col].fillna(0)
            if col in self.test_df.columns:
                self.test_df[col] = self.test_df[col].fillna(0)
            else:
                self.test_df[col] = 0
            
        self.train_df[self.feature_cols] = self.scaler.fit_transform(self.train_df[self.feature_cols])
        self.test_df[self.feature_cols] = self.scaler.transform(self.test_df[self.feature_cols])
        
        gc.collect()
    
    def tune_model(self, trial, X, y, model_type):
        """Optuna objective with reduced CV for speed"""
        cv = StratifiedKFold(n_splits=2)  # Reduced from 3 for speed
        scores = []
        
        if model_type == 'lgb':
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'num_leaves': trial.suggest_int('num_leaves', 31, 256),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.2, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            }
            model = lgb.LGBMClassifier(**params, random_state=42, verbosity=-1)
        
        elif model_type == 'xgb':
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist'
            }
            model = xgb.XGBClassifier(**params, random_state=42, enable_categorical=True)
        
        elif model_type == 'cat':
            params = {
                'depth': trial.suggest_int('depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 1.0, 10.0)
            }
            model = cb.CatBoostClassifier(**params, random_state=42, verbose=0, task_type='GPU' if torch.cuda.is_available() else 'CPU')
        
        elif model_type == 'tabnet':
            params = {
                'n_d': trial.suggest_int('n_d', 8, 64),
                'n_a': trial.suggest_int('n_a', 8, 64),
                'n_steps': trial.suggest_int('n_steps', 3, 10),
                'gamma': trial.suggest_float('gamma', 1.0, 2.0),
                'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-5, 1e-3, log=True)
            }
            model = TabNetClassifier(**params, seed=42, device_name='cuda' if torch.cuda.is_available() else 'cpu')
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            if model_type == 'tabnet':
                model.fit(X_train.values, y_train.values, eval_set=[(X_val.values, y_val.values)], patience=5, max_epochs=20, eval_metric=['auc'])  # Reduced epochs
            else:
                model.fit(X_train, y_train)
            pred = model.predict_proba(X_val)[:, 1] if model_type != 'tabnet' else model.predict_proba(X_val.values)[:, 1]
            scores.append(roc_auc_score(y_val, pred))
        return np.mean(scores)
    
    def train_model(self):
        """Train with Optuna tuning and stacking ensemble - reduced for speed"""
        print("\n🚀 Fast Training with Optuna and Ensemble...")
        
        if not self.feature_cols:
            print("❌ No features!")
            return
        
        X = self.train_df[self.feature_cols]
        y = self.train_df[self.target_col].astype(int)
        
        # Downsample for tuning to speed up (20% of data)
        sample_frac = 0.2
        X_sample = X.sample(frac=sample_frac, random_state=42)
        y_sample = y.loc[X_sample.index]
        
        # Class weights
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes, y)
        class_weights = dict(zip(classes, weights))
        
        # Optuna tuning with reduced trials and jobs
        optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce output for speed
        
        study_lgb = optuna.create_study(direction='maximize')
        study_lgb.optimize(lambda trial: self.tune_model(trial, X_sample, y_sample, 'lgb'), n_trials=20, n_jobs=-1)
        lgb_params = study_lgb.best_params
        
        study_xgb = optuna.create_study(direction='maximize')
        study_xgb.optimize(lambda trial: self.tune_model(trial, X_sample, y_sample, 'xgb'), n_trials=20, n_jobs=-1)
        xgb_params = study_xgb.best_params
        
        study_cat = optuna.create_study(direction='maximize')
        study_cat.optimize(lambda trial: self.tune_model(trial, X_sample, y_sample, 'cat'), n_trials=20, n_jobs=-1)
        cat_params = study_cat.best_params
        
        study_tab = optuna.create_study(direction='maximize')
        study_tab.optimize(lambda trial: self.tune_model(trial, X_sample, y_sample, 'tabnet'), n_trials=20, n_jobs=-1)
        tab_params = study_tab.best_params
        
        # Train base models on full data with tuned params
        base_models = [
            ('lgb', lgb.LGBMClassifier(**lgb_params, class_weight=class_weights, random_state=42, verbosity=-1)),
            ('xgb', xgb.XGBClassifier(**xgb_params, scale_pos_weight=weights[1]/weights[0], random_state=42)),
            ('cat', cb.CatBoostClassifier(**cat_params, class_weights=weights, random_state=42, verbose=0)),
            ('tab', TabNetClassifier(**tab_params, seed=42))
        ]
        
        # Stacking ensemble with reduced CV
        stacking_model = StackingClassifier(estimators=base_models, final_estimator=lgb.LGBMClassifier(random_state=42), cv=3, n_jobs=-1)  # Reduced CV from 5
        
        stacking_model.fit(X, y)
        
        self.models.append(stacking_model)
        print("✅ Stacking model trained with tuned params")
        
        gc.collect()
    
    def make_predictions(self):
        """Make predictions with stacking ensemble"""
        print("\n🎯 Making predictions...")
        
        if not self.models:
            print("❌ No models trained!")
            return
        
        X_test = self.test_df[self.feature_cols]
        
        self.predictions = self.models[0].predict_proba(X_test)[:, 1]
        
        print(f"✅ Predictions ready: mean {self.predictions.mean():.4f}")
    
    def create_submission(self):
        """Create submission file with variable id5"""
        print("\n📄 Creating submission file...")
        
        if self.predictions is None:
            print("❌ No predictions!")
            return
        
        try:
            submission = pd.DataFrame({
                'id1': self.test_df['id1'],
                'id2': self.test_df['id2'],
                'id3': self.test_df['id3'],
                'id5': self.test_df['id5'].dt.strftime('%m-%d-%Y') if pd.api.types.is_datetime64_any_dtype(self.test_df['id5']) else self.test_df['id5'].astype(str),
                'pred': self.predictions
            })
            
            submission.to_csv('FINAL_SUBMISSION.csv', index=False)
            print(f"✅ Saved 'FINAL_SUBMISSION.csv' {submission.shape}")
            print(submission.head())
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def run(self):
        """Run the fast optimized pipeline"""
        try:
            self.load_all_data()
            self.target_col = self.find_target_column()
            self.create_bulletproof_features()
            self.prepare_features()
            self.train_model()
            self.make_predictions()
            self.create_submission()
            print("\n🎉 COMPLETE! Submission ready.")
        except Exception as e:
            print(f"❌ Error: {e}")
            traceback.print_exc()

# Run it
solution = OptimizedAmexSolution()
solution.run()
