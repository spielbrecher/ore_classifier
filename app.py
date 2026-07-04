"""
Финальный Streamlit-дашборд для классификации руд.
Использует RAG-признаки и даёт геологическое объяснение.
"""
import streamlit as st
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os

# Импортируем функции из rag_classification
from rag_classification import (
    load_image_gray,
    segment_to_phases,
    extract_adjacency_features,
)

# ============================================================
# Конфигурация
# ============================================================

st.set_page_config(page_title="Классификатор руд", layout="wide", page_icon="🪨")


# ============================================================
# Умный поиск модели (несколько fallback-путей)
# ============================================================

def find_model_path() -> str:
    """
    Ищет модель в нескольких возможных местах:
    1. models/rag_classifier.joblib (Docker структура)
    2. rag_classification_results_v2/rag_classifier.joblib (локальная разработка)
    3. rag_classifier.joblib (корень проекта)
    4. Переменная окружения MODEL_PATH
    """
    import os

    # Проверяем переменную окружения
    env_path = os.environ.get("MODEL_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # Список возможных путей в порядке приоритета
    candidates = [
        "models/rag_classifier.joblib",
        "rag_classification_results_v2/rag_classifier.joblib",
        "rag_classification_results/rag_classifier.joblib",
        "rag_classifier.joblib",
    ]

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    return candidates[0]  # default


MODEL_PATH = find_model_path()
# MODEL_PATH = "rag_classification_results_v2/rag_classifier.joblib"

CATEGORY_DESCRIPTIONS = {
    "Рядовые": {
        "emoji": "🟢",
        "color": "green",
        "description": "Руда крупными зёрнами, хорошо обособлена от пустой породы. Высокая технологическая ценность.",
        "recommendation": "Стандартная схема флотации. Ожидаемое извлечение >85%.",
    },
    "Труднообогатимые": {
        "emoji": "🔴",
        "color": "red",
        "description": "Руда раздроблена на микроскопические зёрна, сильно замещена нерудными минералами. При дроблении не раскрывается.",
        "recommendation": "Требуется более тонкий помол (<40 мкм) и специальные реагенты. Извлечение <60%.",
    },
    "Оталькованные": {
        "emoji": "🔵",
        "color": "blue",
        "description": "Руда окружена тальком, который обволакивает зёрна и мешает флотации.",
        "recommendation": "Необходима предварительная депрессия талька (карбоксиметилцеллюлоза). Извлечение 50-70%.",
    },
}


# ============================================================
# Загрузка модели
# ============================================================

@st.cache_resource
def load_model():
    """Кешированная загрузка модели."""
    if not Path(MODEL_PATH).exists():
        return None, None
    data = joblib.load(MODEL_PATH)
    return data['model'], data['feature_names']


# ============================================================
# Предсказание
# ============================================================

def predict_ore(gray: np.ndarray, model, feature_names):
    """Делает предсказание для одного изображения."""
    labeled, thresholds = segment_to_phases(gray)
    features, adj_matrix = extract_adjacency_features(labeled, gray)
    
    X = np.array([[features[k] for k in feature_names]])
    prediction = model.predict(X)[0]
    probabilities = dict(zip(model.classes_, model.predict_proba(X)[0]))
    
    return prediction, probabilities, features, labeled, thresholds


# ============================================================
# Визуализации
# ============================================================

def plot_segmentation(gray: np.ndarray, labeled: np.ndarray, thresholds):
    """Визуализация сегментации."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Оригинал
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title('Исходное изображение')
    axes[0].axis('off')
    
    # Цветная сегментация
    colored = np.zeros((*gray.shape, 3), dtype=np.uint8)
    colored[labeled == 0] = [30, 50, 120]   # тёмно-синий = матрица
    colored[labeled == 1] = [150, 150, 150] # серый = тальк
    colored[labeled == 2] = [255, 230, 80]  # жёлтый = руда
    axes[1].imshow(colored)
    axes[1].set_title(f'Сегментация (пороги: {thresholds[0]}, {thresholds[1]})')
    axes[1].axis('off')
    
    # Легенда
    axes[2].axis('off')
    legend_text = """
    **Легенда:**
    
    🟨 **Яркая (жёлтая)** — руда/сульфиды
    
    ⬜ **Серая** — тальк + замещённые зоны
    
    🟦 **Тёмная (синяя)** — пустая порода (матрица)
    """
    axes[2].text(0.1, 0.5, legend_text, fontsize=14, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_key_features(features: dict):
    """Графики ключевых признаков."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Доли фаз
    phases = ['Матрица', 'Тальк', 'Руда']
    values = [features['dark_area_pct'], features['gray_area_pct'], features['bright_area_pct']]
    colors = ['#1e3264', '#999999', '#ffe644']
    axes[0].bar(phases, values, color=colors, edgecolor='black')
    axes[0].set_ylabel('Доля, %')
    axes[0].set_title('Фазовый состав')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(values):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # 2. Морфология руды
    morph_labels = ['Число зёрен', 'Ср. размер', 'Макс. размер']
    morph_values = [
        features['num_bright_regions'],
        features['avg_bright_area'] / 1000,  # в тысячах пикселей
        features['max_bright_area'] / 1000,
    ]
    axes[1].bar(morph_labels, morph_values, color='orange', edgecolor='black')
    axes[1].set_title('Морфология зёрен руды')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(morph_values):
        axes[1].text(i, v + 0.02 * max(morph_values), f'{v:.2f}', 
                    ha='center', fontweight='bold', fontsize=9)
    
    # 3. Контакты
    contact_labels = ['Руда↔Матрица', 'Руда↔Тальк', 'Руда↔Руда']
    contact_values = [
        features['ore_matrix_contact'] * 100,
        features['ore_talc_contact'] * 100,
        features['ore_ore_contact'] * 100,
    ]
    axes[2].bar(contact_labels, contact_values, color=['green', 'red', 'gold'], edgecolor='black')
    axes[2].set_ylabel('% контактов руды')
    axes[2].set_title('Контакты руды с другими фазами')
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(contact_values):
        axes[2].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    return fig


# ============================================================
# Основной интерфейс
# ============================================================

def main():
    st.title("🪨 Классификатор типов руд")
    st.caption("Автоматический анализ полированных шлифов на основе RAG-признаков")
    
    # Загрузка модели
    model, feature_names = load_model()
    if model is None:
        st.error(f"Модель не найдена по пути: {MODEL_PATH}")
        st.info("Сначала запустите `python rag_classification.py` для обучения модели.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ О системе")
        st.markdown("""
        **Что анализируется:**
        - Доли минеральных фаз
        - Размер и форма зёрен руды
        - Контакты между фазами
        - Пространственное распределение
        
        **Точность модели:** ~73-82% (5-fold CV)
        """)
        
        st.divider()
        st.header("📖 Легенда")
        for cat, info in CATEGORY_DESCRIPTIONS.items():
            st.markdown(f"**{info['emoji']} {cat}**")
            st.caption(info['description'][:80] + "...")
    
    # Загрузка изображения
    st.header("📤 Загрузите изображение шлифа")
    uploaded_file = st.file_uploader(
        "Выберите файл (JPG, PNG, TIFF)",
        type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
    )
    
    if uploaded_file is not None:
        # Загрузка
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            gray = load_image_gray(tmp_path)
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")
            os.unlink(tmp_path)
            return
        
        # Масштабирование для скорости
        if max(gray.shape) > 1500:
            scale = 1500 / max(gray.shape)
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
        
        st.info(f"📐 Размер изображения: {gray.shape[1]}×{gray.shape[0]} px")
        
        # Предсказание
        with st.spinner("Анализ изображения..."):
            prediction, probabilities, features, labeled, thresholds = predict_ore(
                gray, model, feature_names
            )
        
        # === РЕЗУЛЬТАТ ===
        st.divider()
        st.header("🎯 Результат классификации")
        
        info = CATEGORY_DESCRIPTIONS[prediction]
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style='text-align: center; padding: 30px; 
                        background-color: {info['color']}; color: white; 
                        border-radius: 15px; font-size: 28px;'>
                {info['emoji']} <b>{prediction.upper()}</b>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"### 📝 Описание\n{info['description']}")
        st.markdown(f"### 💡 Рекомендация\n{info['recommendation']}")
        
        # Вероятности
        st.subheader("📊 Уверенность модели")
        prob_cols = st.columns(3)
        for i, (cat, prob) in enumerate(sorted(probabilities.items(), key=lambda x: x[1], reverse=True)):
            cat_info = CATEGORY_DESCRIPTIONS[cat]
            with prob_cols[i]:
                st.metric(
                    f"{cat_info['emoji']} {cat}",
                    f"{prob * 100:.1f}%",
                    delta="✓" if cat == prediction else "",
                )
        
        # === ВИЗУАЛИЗАЦИИ ===
        st.divider()
        st.header("🔍 Детальный анализ")
        
        tab1, tab2, tab3 = st.tabs(["📷 Сегментация", "📈 Признаки", "🔬 Матрица смежности"])
        
        with tab1:
            fig = plot_segmentation(gray, labeled, thresholds)
            st.pyplot(fig)
            plt.close(fig)
        
        with tab2:
            fig = plot_key_features(features)
            st.pyplot(fig)
            plt.close(fig)
            
            # Таблица всех признаков
            with st.expander("📋 Все признаки"):
                feature_df = {k: [f"{v:.4f}"] for k, v in sorted(features.items())}
                st.dataframe(feature_df, use_container_width=True)
        
        with tab3:
            st.markdown("""
            **Матрица смежности фаз** показывает, как часто пиксели разных фаз 
            граничат друг с другом. Нормализована на общее число контактов.
            """)
            
            # Пересчитываем матрицу для визуализации
            from rag_classification import extract_adjacency_features
            _, adj_matrix = extract_adjacency_features(labeled, gray)
            
            if adj_matrix.sum() > 0:
                adj_norm = adj_matrix / adj_matrix.sum()
            else:
                adj_norm = adj_matrix
            
            fig, ax = plt.subplots(figsize=(8, 6))
            phase_names = ['Матрица', 'Тальк', 'Руда']
            im = ax.imshow(adj_norm, cmap='YlOrRd', vmin=0, vmax=0.15)
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(phase_names, fontsize=11)
            ax.set_yticklabels(phase_names, fontsize=11)
            ax.set_title('Матрица смежности фаз', fontsize=13)
            
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{adj_norm[i,j]:.3f}',
                           ha='center', va='center', fontsize=12,
                           color='white' if adj_norm[i,j] > 0.07 else 'black')
            
            plt.colorbar(im, ax=ax, label='Доля контактов')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        # Очистка
        os.unlink(tmp_path)
    
    else:
        # Примеры
        st.info("👆 Загрузите изображение шлифа для анализа")
        
        st.divider()
        st.header("💡 Что определяет тип руды")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("**🟢 Рядовая**")
            st.markdown("""
            - Крупные зёрна руды
            - Много руды (>20%)
            - Руда изолирована от матрицы
            - Мало контактов с тальком
            """)
        
        with col2:
            st.error("**🔴 Труднообогатимая**")
            st.markdown("""
            - Мелкие зёрна руды
            - Мало руды (<15%)
            - Сильная замещённость
            - Много зёрен (>3000)
            """)
        
        with col3:
            st.info("**🔵 Оталькованная**")
            st.markdown("""
            - Много талька
            - Руда окружена тальком (>80% контактов)
            - Низкий индекс изоляции
            """)


if __name__ == "__main__":
    main()