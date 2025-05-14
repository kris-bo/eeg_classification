import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from mne import Epochs, pick_types
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.preprocessing import ICA
from mne.channels import make_standard_montage
from mne.decoding import CSP
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import os
import gc
from time import time

mne.set_log_level('WARNING')

subject = list(range(40, 90))
run_imagery = [4, 6, 8, 10, 12, 14]
data_dir = "/Users/kris/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0"
raw_files = []
target_sfreq = 160

for person_number in subject:
    for j in run_imagery:
        subject_dir = f"S{person_number:03d}"
        file_path = os.path.join(data_dir, subject_dir, f"S{person_number:03d}R{j:02d}.edf")

        if not os.path.exists(file_path):
            print(f"Skipping {file_path}, file not found.")
            continue

        raw_imagery = read_raw_edf(file_path, preload=True, stim_channel='auto')

        if raw_imagery.info['sfreq'] != target_sfreq:
            print(f"Resampling {file_path} from {raw_imagery.info['sfreq']} Hz to {target_sfreq} Hz")
            raw_imagery.resample(target_sfreq)

        events, event_id = mne.events_from_annotations(raw_imagery)

        if j in [4, 8, 12]:
            mapping = {1: 'rest', 2: 'imagine/hands', 3: 'imagine/hands'}
        elif j in [6, 10, 14]:
            mapping = {1: 'rest', 2: 'imagine/hands', 3: 'imagine/feet'}

        if len(events) > 0:
            annot_from_events = mne.annotations_from_events(
                events, event_desc=mapping, 
                sfreq=raw_imagery.info['sfreq'],
                orig_time=raw_imagery.info['meas_date']
            )
            raw_imagery.set_annotations(annot_from_events)

        raw_files.append(raw_imagery)

if raw_files:
    raw = concatenate_raws(raw_files)

eegbci.standardize(raw)
montage = make_standard_montage('standard_1005')
raw.set_montage(montage)
raw.filter(5., 40., fir_design='firwin', skip_by_annotation='edge')

event_id = {'imagine/hands': 1, 'imagine/feet': 2}
tmin, tmax = 0., 2.

def run_ica(raw, method='fastica', fit_params=None):
    print(f"Running ICA with method: {method}")
    raw_corrected = raw.copy()
    raw_corrected.resample(50)
    picks = pick_types(raw_corrected.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    ica = ICA(n_components=15, method=method, fit_params=fit_params, random_state=97)
    t0 = time()
    ica.fit(raw_corrected, picks=picks)
    fit_time = time() - t0
    print(f"ICA fitted in {fit_time:.1f} seconds.")
    ica.plot_components(title=f"ICA using {method}")
    plt.show()
    eog_indices, scores = ica.find_bads_eog(raw_corrected, ch_name='Fpz', threshold=1.5)
    ica.exclude.extend(eog_indices)
    raw_corrected = ica.apply(raw_corrected, exclude=ica.exclude)
    print("Excluded components:", ica.exclude)
    return raw_corrected

raw_fastica = run_ica(raw, 'fastica')
print("after run_ica")

events, event_dict = mne.events_from_annotations(raw, event_id=event_id)
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
print(epochs.get_data().shape)
print(f"Number of epochs: {len(epochs)}")
epochs.drop_bad(reject=dict(eeg=60e-6))
print(f"Number of epochs: {len(epochs)}")

labels = epochs.events[:, -1] - 1
print(labels)

cv = ShuffleSplit(10, test_size=0.4, random_state=42)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42)
}

X = epochs.get_data().astype(np.float64)
labels = epochs.events[:, -1] - 1
print(f"X shape before CSP: {X.shape}, dtype: {X.dtype}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_score = 0
best_clf = None

for clf_name, clf in classifiers.items():
    csp = CSP(n_components=6, transform_into='average_power')
    pipeline = make_pipeline(csp, clf)
    scores = cross_val_score(pipeline, X, labels, cv=cv, n_jobs=1, error_score="raise")
    print(f"Classifier: {clf_name}, Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_clf = pipeline
    del pipeline, csp, clf, scores
    gc.collect()

best_clf.fit(X, labels)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
print("\nepoch nb: [prediction] [truth] equal?")
all_preds = np.zeros_like(labels)

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, labels)):
    best_clf.fit(X[train_idx], labels[train_idx])
    preds = best_clf.predict(X[test_idx])
    all_preds[test_idx] = preds

for i, (pred, true) in enumerate(zip(all_preds, labels)):
    is_equal = pred == true
    print(f"epoch {i:02d}: [{pred}] [{true}] {is_equal}")

sklearn_pca = PCA(n_components=2)
X_pca = sklearn_pca.fit_transform(X.reshape(len(labels), -1))

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolors='k')
plt.colorbar(label='Imagined Movement Type')
plt.title('PCA Projection of EEG Data')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
exit()
