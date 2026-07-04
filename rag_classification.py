"""
RAG-КЛАССИФИКАЦИЯ РУД (улучшенная версия)
- Передача gray для расчёта интенсивностей
- Улучшенная сегментация (сужение "серой" фазы)
- Дополнительные признаки: число зёрен, средняя площадь, интенсивности
"""
import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import label as ndlabel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import joblib

plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']


def load_image_gray(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        img_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)


# ============================================================
# 1. СЕГМЕНТАЦИЯ НА ТРИ ФАЗЫ (УЛУЧШЕННАЯ)
# ============================================================

def segment_to_phases(gray: np.ndarray) -> tuple:
    """
    Улучшенная сегментация: сужаем "серую" фазу,
    чтобы тальк не "захватывал" половину изображения.
    """
    from skimage.filters import threshold_multiotsu
    thresholds = threshold_multiotsu(gray, classes=3)
    t_low, t_high = thresholds
    
    # Сужаем диапазон серой фазы
    gray_range = t_high - t_low
    t_low_adj = int(t_low + gray_range * 0.25)    # сдвигаем нижний порог вверх
    t_high_adj = int(t_high - gray_range * 0.15)  # сдвигаем верхний порог вниз
    
    # Защита от пустых фаз
    if t_low_adj >= t_high_adj:
        t_low_adj = t_low
        t_high_adj = t_high
    
    labeled = np.zeros_like(gray, dtype=np.int32)
    labeled[gray >= t_low_adj] = 1
    labeled[gray >= t_high_adj] = 2
    
    return labeled, (t_low_adj, t_high_adj)


# ============================================================
# 2. ИЗВЛЕЧЕНИЕ RAG-ПРИЗНАКОВ (ИСПРАВЛЕНО: теперь принимает gray)
# ============================================================

def extract_adjacency_features(labeled: np.ndarray, gray: np.ndarray) -> dict:
    """
    Извлекает RAG-признаки + интенсивности + морфологию.
    """
    features = {}
    
    # === 1. Доли фаз ===
    total = labeled.size
    features['dark_area_pct'] = (labeled == 0).sum() / total * 100
    features['gray_area_pct'] = (labeled == 1).sum() / total * 100
    features['bright_area_pct'] = (labeled == 2).sum() / total * 100
    
    # === 2. Средняя яркость каждой фазы (ИСПРАВЛЕНО) ===
    features['dark_mean_intensity'] = gray[labeled == 0].mean() if (labeled == 0).any() else 0
    features['gray_mean_intensity'] = gray[labeled == 1].mean() if (labeled == 1).any() else 0
    features['bright_mean_intensity'] = gray[labeled == 2].mean() if (labeled == 2).any() else 0
    
    # Стандартные отклонения яркости внутри фаз
    features['dark_std_intensity'] = gray[labeled == 0].std() if (labeled == 0).sum() > 1 else 0
    features['gray_std_intensity'] = gray[labeled == 1].std() if (labeled == 1).sum() > 1 else 0
    features['bright_std_intensity'] = gray[labeled == 2].std() if (labeled == 2).sum() > 1 else 0
    
    # === 3. Матрица смежности фаз ===
    adjacency_matrix = np.zeros((3, 3), dtype=np.int64)
    
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4 соседа (быстрее и надёжнее)
    
    for dy, dx in offsets:
        shifted = np.roll(np.roll(labeled, dy, axis=0), dx, axis=1)
        boundary_mask = labeled != shifted
        
        if boundary_mask.any():
            p1 = labeled[boundary_mask]
            p2 = shifted[boundary_mask]
            for i in range(len(p1)):
                adjacency_matrix[p1[i], p2[i]] += 1
    
    adjacency_matrix = adjacency_matrix // 2
    total_boundary = adjacency_matrix.sum()
    
    adj_norm = adjacency_matrix / total_boundary if total_boundary > 0 else adjacency_matrix
    
    for i in range(3):
        for j in range(3):
            features[f'adj_{i}_{j}'] = adj_norm[i, j]
    
    # === 4. Геологические признаки ===
    bright_contacts = adj_norm[2, :].sum()
    if bright_contacts > 0:
        features['ore_talc_contact'] = adj_norm[2, 1] / bright_contacts
        features['ore_matrix_contact'] = adj_norm[2, 0] / bright_contacts
        features['ore_ore_contact'] = adj_norm[2, 2] / bright_contacts
    else:
        features['ore_talc_contact'] = 0
        features['ore_matrix_contact'] = 0
        features['ore_ore_contact'] = 0
    
    features['ore_isolation_index'] = features['ore_ore_contact'] - features['ore_talc_contact']
    features['boundary_density'] = total_boundary / total
    
    # === 5. Морфологические признаки (число зёрен) ===
    _, num_dark = ndlabel(labeled == 0)
    _, num_gray = ndlabel(labeled == 1)
    _, num_bright = ndlabel(labeled == 2)
    
    features['num_dark_regions'] = num_dark
    features['num_gray_regions'] = num_gray
    features['num_bright_regions'] = num_bright
    
    # Средняя площадь зёрен каждой фазы
    features['avg_bright_area'] = (labeled == 2).sum() / max(num_bright, 1)
    features['avg_gray_area'] = (labeled == 1).sum() / max(num_gray, 1)
    features['avg_dark_area'] = (labeled == 0).sum() / max(num_dark, 1)
    
    # Крупность зёрен руды (большие зёрна = лучше для обогащения)
    if num_bright > 0:
        bright_areas = ndimage.sum(labeled == 2, labeled == 2, range(1, num_bright + 1))
        features['max_bright_area'] = max(bright_areas) if len(bright_areas) > 0 else 0
        features['bright_area_cv'] = np.std(bright_areas) / (np.mean(bright_areas) + 1e-6)
    else:
        features['max_bright_area'] = 0
        features['bright_area_cv'] = 0
    
    # Плотность зёрен руды на 10000 пикселей
    features['ore_grain_density'] = num_bright / (total / 10000)
    
    return features, adjacency_matrix


# ============================================================
# 3. ПОСТРОЕНИЕ ДАТАСЕТА
# ============================================================

def build_dataset(dataset_path: str, samples_per_cat: int = 15) -> tuple:
    categories = {}
    for root, dirs, files in os.walk(dataset_path):
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.tif', '.jpeg', '.tiff'))]
        if image_files:
            cat_name = Path(root).name
            if cat_name not in ["Области оталькования"]:
                categories[cat_name] = [os.path.join(root, f) for f in image_files]
    
    X, y, meta = [], [], []
    feature_names = None
    
    for cat_name, img_paths in categories.items():
        print(f"📂 {cat_name}")
        samples = random.sample(img_paths, min(samples_per_cat, len(img_paths)))
        
        for img_path in samples:
            try:
                gray = load_image_gray(img_path)
                if max(gray.shape) > 1024:
                    scale = 1024 / max(gray.shape)
                    gray = cv2.resize(gray, None, fx=scale, fy=scale)
                
                labeled, _ = segment_to_phases(gray)
                
                # ИСПРАВЛЕНО: передаём gray
                features, adj_matrix = extract_adjacency_features(labeled, gray)
                
                if feature_names is None:
                    feature_names = sorted(features.keys())
                
                feature_vector = [features[k] for k in feature_names]
                
                X.append(feature_vector)
                y.append(cat_name)
                meta.append({
                    'filename': Path(img_path).name,
                    'category': cat_name,
                    'adj_matrix': adj_matrix,
                    'features': features,
                })
                
                print(f"   ✓ {Path(img_path).name}: "
                      f"руда-тальк={features['ore_talc_contact']:.3f}, "
                      f"руда={features['bright_area_pct']:.1f}%, "
                      f"зёрен руды={features['num_bright_regions']}")
                
            except Exception as e:
                print(f"   ✗ {Path(img_path).name}: {e}")
                import traceback
                traceback.print_exc()
    
    return np.array(X), np.array(y), meta, feature_names


# ============================================================
# 4. ОБУЧЕНИЕ И ОЦЕНКА
# ============================================================

def train_and_evaluate(X, y, feature_names):
    print("\n" + "="*70)
    print("🔬 ОБУЧЕНИЕ КЛАССИФИКАТОРА НА УЛУЧШЕННЫХ RAG-ПРИЗНАКАХ")
    print("="*70)
    print(f"Всего признаков: {len(feature_names)}")
    print(f"Всего образцов: {len(X)}")
    
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=12, 
        min_samples_split=3, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"\n🎯 5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Если мало данных — оставляем leave-one-out
    if len(X) < 30:
        loo_scores = cross_val_score(clf, X, y, cv=len(X), scoring='accuracy')
        print(f"🎯 Leave-One-Out accuracy: {loo_scores.mean():.3f} ± {loo_scores.std():.3f}")
    
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("\n📊 Confusion matrix (train set, upper bound):")
    print(classification_report(y, y_pred))
    
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    
    print("🏆 Топ-20 важных признаков:")
    for idx in top_idx:
        print(f"   {importances[idx]:.4f}  {feature_names[idx]}")
    
    return clf, importances, feature_names


# ============================================================
# 5. ВИЗУАЛИЗАЦИИ
# ============================================================

def visualize_adjacency_matrices(meta: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    categories = sorted(list(set(m['category'] for m in meta)))
    phase_names = ['Тёмная\n(матрица)', 'Серая\n(тальк)', 'Яркая\n(руда)']
    
    fig, axes = plt.subplots(1, len(categories), figsize=(5*len(categories), 4.5))
    if len(categories) == 1:
        axes = [axes]
    
    for ax, cat in zip(axes, categories):
        cat_meta = [m for m in meta if m['category'] == cat]
        avg_adj = np.mean([m['adj_matrix'] for m in cat_meta], axis=0)
        
        if avg_adj.sum() > 0:
            avg_adj = avg_adj / avg_adj.sum()
        
        im = ax.imshow(avg_adj, cmap='YlOrRd', vmin=0, vmax=0.15)
        ax.set_title(f'{cat}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(phase_names, fontsize=8)
        ax.set_yticklabels(phase_names, fontsize=8)
        
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{avg_adj[i,j]:.3f}',
                       ha='center', va='center', fontsize=9,
                       color='white' if avg_adj[i,j] > 0.07 else 'black')
        
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Матрицы смежности фаз (нормализованные)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adjacency_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Матрицы сохранены: {output_dir}/adjacency_matrices.png")


def visualize_2d_space(X, y, feature_names, importances, output_dir):
    """PCA + выделение топ-2 признаков для scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    colors = {'Рядовые': 'green', 'Труднообогатимые': 'red', 'Оталькованные': 'blue'}
    markers = {'Рядовые': 'o', 'Труднообогатимые': 's', 'Оталькованные': '^'}
    
    for cat in set(y):
        mask = y == cat
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=colors.get(cat, 'gray'),
                       marker=markers.get(cat, 'o'),
                       s=100, alpha=0.7, label=cat, edgecolors='black')
    
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0].set_title('PCA-проекция всех признаков')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Scatter по топ-2 признакам
    top_2 = np.argsort(importances)[::-1][:2]
    f1_idx, f2_idx = top_2
    f1_name, f2_name = feature_names[f1_idx], feature_names[f2_idx]
    
    for cat in set(y):
        mask = y == cat
        axes[1].scatter(X[mask, f1_idx], X[mask, f2_idx],
                       c=colors.get(cat, 'gray'),
                       marker=markers.get(cat, 'o'),
                       s=100, alpha=0.7, label=cat, edgecolors='black')
    
    axes[1].set_xlabel(f1_name)
    axes[1].set_ylabel(f2_name)
    axes[1].set_title(f'Топ-2 признака: {f1_name} vs {f2_name}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_space.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ PCA + scatter сохранены: {output_dir}/feature_space.png")


def compare_categories(meta: list, output_dir: str):
    """Сравнивает средние значения ключевых признаков по категориям."""
    categories = sorted(list(set(m['category'] for m in meta)))
    
    key_features = [
        'bright_area_pct', 'gray_area_pct',
        'ore_talc_contact', 'ore_isolation_index',
        'num_bright_regions', 'avg_bright_area',
        'bright_mean_intensity',
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    
    colors = {'Рядовые': 'green', 'Труднообогатимые': 'red', 'Оталькованные': 'blue'}
    
    for i, feat in enumerate(key_features):
        if i >= len(axes):
            break
        
        ax = axes[i]
        means = []
        stds = []
        
        for cat in categories:
            cat_values = [m['features'][feat] for m in meta if m['category'] == cat]
            means.append(np.mean(cat_values))
            stds.append(np.std(cat_values))
        
        ax.bar(categories, means, yerr=stds, capsize=5,
               color=[colors.get(c, 'gray') for c in categories],
               edgecolor='black', alpha=0.8)
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.set_ylabel(feat)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=15)
        
        for j, (m, s) in enumerate(zip(means, stds)):
            ax.text(j, m + s + 0.02 * max(means), f'{m:.2f}', 
                   ha='center', fontsize=9, fontweight='bold')
    
    for i in range(len(key_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Сравнение категорий по ключевым признакам', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Сравнение категорий: {output_dir}/category_comparison.png")


# ============================================================
# MAIN
# ============================================================

def main():
    DATASET_PATH = "dataset"
    OUTPUT_DIR = "rag_classification_results_v2"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("RAG-КЛАССИФИКАЦИЯ РУД (улучшенная версия)")
    print("="*70)
    
    X, y, meta, feature_names = build_dataset(DATASET_PATH, samples_per_cat=15)
    
    print(f"\n✅ Собрано образцов: {len(X)}")
    
    if len(X) == 0:
        print("❌ Нет данных для обучения!")
        return
    
    clf, importances, feature_names = train_and_evaluate(X, y, feature_names)
    
    # Сохраняем модель
    model_path = os.path.join(OUTPUT_DIR, 'rag_classifier.joblib')
    joblib.dump({
        'model': clf,
        'feature_names': feature_names,
    }, model_path)
    print(f"\n💾 Модель сохранена: {model_path}")
    
    print("\n🎨 Визуализации...")
    visualize_adjacency_matrices(meta, OUTPUT_DIR)
    visualize_2d_space(X, y, feature_names, importances, OUTPUT_DIR)
    compare_categories(meta, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("🎯 ЧТО ИСКАТЬ В РЕЗУЛЬТАТАХ")
    print("="*70)
    print("""
1. adj_2_1 (руда↔тальк): высокое в "Оталькованные"
2. bright_area_pct (доля руды): высокое в "Рядовые"
3. num_bright_regions (число зёрен руды): много мелких vs мало крупных
4. ore_isolation_index: положительный = руда изолирована
5. max_bright_area: крупные зёрна руды → хорошее обогащение

Откройте category_comparison.png — там видны различия между категориями!
    """)


if __name__ == "__main__":
    main()