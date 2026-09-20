
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import warnings
from deap import base, creator, tools, algorithms
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import random
import random
warnings.filterwarnings('ignore')

print("="*60)
print("EMJM - QSAR PIPELINE WITH REAL ChEMBL DATA (FIXED)")
print("="*60)

# -----------------------------
# 1. LOAD REAL DATA FROM ChEMBL
# -----------------------------
def load_real_chembl_data(target_chembl_id="CHEMBL203", max_records=150):
    print(f"\nFetching real data from ChEMBL for target {target_chembl_id}...")
    activity = new_client.activity
    data = activity.filter(target_chembl_id=target_chembl_id, 
                          standard_type="IC50",
                          standard_relation="=")
    
    records = []
    count = 0
    for d in data:
        if count >= max_records:
            break
        try:
            smi = d.get("canonical_smiles")
            val = d.get("standard_value")
            if not smi or not val:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            val = float(val)
            if val <= 0:
                continue
            pIC50 = -np.log10(val * 1e-9)
            if 0 < pIC50 < 15:
                records.append([smi, pIC50])
                count += 1
        except:
            continue
    
    df = pd.DataFrame(records, columns=["SMILES", "pIC50"]).drop_duplicates(subset="SMILES")
    print(f"Loaded {len(df)} unique molecules with valid pIC50 values.")
    return df

# -----------------------------
# 2. CONFORMER GENERATION
# -----------------------------
class ConformerGenerator:
    def __init__(self, num_conformers=20, optimize_ff=True):
        self.num_conformers = num_conformers
        self.optimize_ff = optimize_ff
    
    def generate_conformers(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        try:
            # Embed multiple conformers
            AllChem.EmbedMultipleConfs(mol, numConfs=self.num_conformers, randomSeed=42)
            if self.optimize_ff:
                # Optimize each conformer with MMFF
                for conf_id in range(mol.GetNumConformers()):
                    try:
                        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
                    except:
                        pass
            return mol
        except:
            return None

# -----------------------------
# 3. ELECTRON-CONFORMATIONAL FEATURES (NO MMFF ERROR)
# -----------------------------
class ElectronConformationalFeatures:
    def __init__(self):
        self.feature_names = []
    
    def calculate_quantum_features(self, mol, conf_id=0):
        features = {}
        if mol is None or mol.GetNumConformers() == 0:
            return self._empty_features()
        
        # Try to get force field energy (robust way)
        energy = 0.0
        try:
            # First sanitize for MMFF
            AllChem.MMFFSanitizeMolecule(mol)
            # Get force field (using positional arguments to avoid version issues)
            ff = AllChem.MMFFGetMoleculeForceField(mol, confId=conf_id)
            if ff is not None:
                energy = ff.CalcEnergy()
        except:
            energy = 0.0
        
        features['conformer_energy'] = energy
        features['num_atoms'] = mol.GetNumAtoms()
        features['num_bonds'] = mol.GetNumBonds()
        features['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
        features['mol_weight'] = Descriptors.MolWt(mol)
        features['logP'] = Descriptors.MolLogP(mol)
        features['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
        features['num_h_donors'] = Descriptors.NumHDonors(mol)
        features['num_h_acceptors'] = Descriptors.NumHAcceptors(mol)
        
        # Gasteiger charges
        AllChem.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            try:
                charges.append(float(atom.GetProp('_GasteigerCharge')))
            except:
                charges.append(0)
        features['max_charge'] = max(charges) if charges else 0
        features['min_charge'] = min(charges) if charges else 0
        features['mean_charge'] = np.mean(charges) if charges else 0
        features['charge_range'] = features['max_charge'] - features['min_charge']
        return features
    
    def _empty_features(self):
        keys = ['conformer_energy', 'num_atoms', 'num_bonds', 'num_heavy_atoms',
                'mol_weight', 'logP', 'num_rotatable_bonds', 'num_h_donors',
                'num_h_acceptors', 'max_charge', 'min_charge', 'mean_charge', 'charge_range']
        return {k: 0 for k in keys}
    
    def extract_all(self, mol):
        if mol is None:
            return self._empty_features()
        all_features = []
        num_confs = min(mol.GetNumConformers(), 10)
        for conf_id in range(num_confs):
            features = self.calculate_quantum_features(mol, conf_id)
            all_features.append(list(features.values()))
        if all_features:
            mean_features = np.mean(all_features, axis=0)
        else:
            mean_features = [0]*13
        feature_keys = list(self.calculate_quantum_features(mol).keys())
        return dict(zip(feature_keys, mean_features))

# -----------------------------
# 4. FINGERPRINT + ECM FEATURES
# -----------------------------
def morgan_fingerprint(smiles, nBits=2048, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits))

def extract_all_features(df, use_ecm=True):
    print("\nExtracting features from molecules...")
    X_fp = np.array([morgan_fingerprint(s) for s in df.SMILES])
    
    if use_ecm:
        conf_gen = ConformerGenerator(num_conformers=15, optimize_ff=True)  # reduced for speed
        ecm_extractor = ElectronConformationalFeatures()
        ecm_features = []
        for i, smi in enumerate(df.SMILES):
            if i % 20 == 0:
                print(f"  Progress: {i}/{len(df)}")
            mol = conf_gen.generate_conformers(smi)
            if mol:
                feat = ecm_extractor.extract_all(mol)
                ecm_features.append(list(feat.values()))
            else:
                ecm_features.append([0]*13)
        X_ecm = np.array(ecm_features)
        X_combined = np.hstack([X_fp, X_ecm])
        print(f"Final features: {X_combined.shape[1]} dimensions (2048 FP + {X_ecm.shape[1]} ECM)")
        return X_combined
    else:
        return X_fp

# -----------------------------
# 5. GENETIC FEATURE SELECTION (OPTIONAL)
# -----------------------------
class GeneticFeatureSelector:
    def __init__(self, population_size=20, generations=10, cv_folds=5):
        self.population_size = population_size
        self.generations = generations
        self.cv_folds = cv_folds
    
    def select_features(self, X, y, model):
        print("\nRunning genetic algorithm for feature selection...")
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
            del creator.Individual
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate, X=X, y=y, model=model)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        pop = toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=self.generations, 
                           stats=stats, halloffame=hof, verbose=False)
        best = hof[0]
        selected = [i for i, bit in enumerate(best) if bit == 1]
        print(f"Selected {len(selected)} features out of {X.shape[1]}")
        return selected
    
    def _evaluate(self, individual, X, y, model):
        selected = [i for i, bit in enumerate(individual) if bit == 1]
        if len(selected) < 5:
            return (0,)
        X_sel = X[:, selected]
        scores = cross_val_score(model, X_sel, y, cv=self.cv_folds, scoring='r2')
        return (scores.mean(),)

# -----------------------------
# 6. ENSEMBLE MODEL
# -----------------------------
class EGFRModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.selected_features = None
    
    def train(self, X_train, y_train, use_ga=True):
        X_scaled = self.scaler.fit_transform(X_train)
        if use_ga and X_scaled.shape[1] > 100:
            base_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
            ga = GeneticFeatureSelector(population_size=15, generations=8)
            self.selected_features = ga.select_features(X_scaled, y_train, base_model)
            X_train_sel = X_scaled[:, self.selected_features]
        else:
            X_train_sel = X_scaled
        
        self.models['xgb'] = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, 
                                          subsample=0.8, random_state=42, verbosity=0)
        self.models['rf'] = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        self.models['gbr'] = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
        
        for name, m in self.models.items():
            m.fit(X_train_sel, y_train)
        return self
    
    def predict(self, X_test):
        X_scaled = self.scaler.transform(X_test)
        if self.selected_features is not None:
            X_test_sel = X_scaled[:, self.selected_features]
        else:
            X_test_sel = X_scaled
        preds = [m.predict(X_test_sel) for m in self.models.values()]
        weights = [0.4, 0.3, 0.3]
        return np.average(preds, weights=weights, axis=0)

# -----------------------------
# 7. MAIN PIPELINE
# -----------------------------
def main():
    print("\n" + "="*60)
    print("Model for EGFR pIC50 Prediction Using Scaffold-Based Data Splitting")
    print("="*60)
    
    # Load real data (target: EGFR CHEMBL203)
    df = load_real_chembl_data(target_chembl_id="CHEMBL203", max_records=120)
    
    if len(df) < 20:
        print("Insufficient data. Try a different target_chembl_id.")
        return None
    
    print(f"\nDataset: {len(df)} molecules")
    print(f"pIC50 range: min={df.pIC50.min():.2f}, max={df.pIC50.max():.2f}, mean={df.pIC50.mean():.2f}")
    
# ========== اضافه کردن PCA برای کاهش ویژگی‌ها ==========
    from sklearn.decomposition import PCA

# بعد از استخراج X و y
    X = extract_all_features(df, use_ecm=True)
    y = df.pIC50.values

# کاهش ابعاد به 50 ویژگی اصلی
    pca = PCA(n_components=50)
    X = pca.fit_transform(X)
    print("The features were reduced to 50 dimensions using Principal Component Analysis (PCA).")
# ======================================================

# سپس ادامه کد (تقسیم داده، GA، مدل) بدون تغییر    
    # ========== SCAFFOLD SPLIT (جایگزین train_test_split) ==========
    def get_scaffold(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

    # Group molecules by scaffold
# ========== RANDOM SPLIT با درصد ==========
    from sklearn.model_selection import train_test_split

    test_size = 19   # یا 0.15 برای درصد
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=42
     )

    print(f"Random split: {len(X_train)} train, {len(X_test)} test")
# ==================================# =================================================    
    print(f"Scaffold split: {len(X_train)} train, {len(X_test)} test")
    # ================================================================    
    model = EGFRModel()
    model.train(X_train, y_train, use_ga=True)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\n" + "="*60)
    print("Evaluation Results on Real Data:")
    print("="*60)
    print(f"R²:  {r2:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    
    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].scatter(y_test, y_pred, alpha=0.6)
    ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax[0].set_xlabel("True pIC50")
    ax[0].set_ylabel("Predicted pIC50")
    ax[0].set_title(f"EGFR Predictions (R² = {r2:.3f})")
    
    residuals = y_test - y_pred
    ax[1].hist(residuals, bins=10, edgecolor='black')
    ax[1].set_xlabel("Residuals")
    ax[1].set_ylabel("Frequency")
    ax[1].set_title("Residual Distribution")
    plt.tight_layout()
    plt.savefig("EMFR_real_results.png", dpi=300)
    plt.show()
    
    joblib.dump(model, "EMFR_model_real.pkl")
    print("\nModel saved: EMFR_model_real.pkl")
    return model

# -----------------------------
# 8. RUN
# -----------------------------
if __name__ == "__main__":
    main()
